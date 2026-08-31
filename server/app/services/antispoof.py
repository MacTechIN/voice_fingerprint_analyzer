"""오디오 딥페이크 탐지 (Anti-Spoofing, Phase 8).

TTS·보이스 컨버전으로 만든 합성 음성이 화자 인증을 우회하는 논리적 접근(Logical
Access) 공격을 막는다 (02 §5, FR-16).

**화자 인증과 다른 문제를 푼다.** 화자 인증은 "이 목소리가 등록된 사람인가"를
묻고, 안티스푸핑은 "이 소리가 사람이 실제로 낸 것인가"를 묻는다. 잘 만든 합성
음성은 화자 인증을 통과하므로 — 그것이 공격의 목적이다 — 별도 방어가 필요하다.

## 모델

`clovaai/aasist`의 **AASIST-L** (MIT). 파라미터 85K로 가볍고 CPU 실시간이 가능해
전체 요청에 적용할 수 있다. 원시 파형을 직접 받아 시간·주파수 도메인의 미세한
인공 아티팩트를 그래프 어텐션으로 포착한다.

08 §4의 2단 캐스케이드 설계에서 **1차 스크리닝**에 해당한다. 2차 정밀 판정
(XLSR+AASIST)은 SSL 프론트엔드가 300M 파라미터라 CPU 지연 예산을 넘어 미착수다.

## 점수의 방향

모델은 2개 로짓을 낸다: `[spoof, bonafide]`. 본 모듈은 이를 **spoof 확률**로
바꿔 반환한다 — 값이 클수록 합성 음성일 가능성이 높다. 임계값을 넘으면 차단한다.
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_model = None
_load_lock = threading.Lock()

#: AASIST가 학습된 입력 길이 (64600 샘플 = 16kHz에서 약 4.04초).
#: 이보다 짧으면 반복해 채우고, 길면 앞에서 자른다 — 원 저자 구현과 같은 방식이다.
NB_SAMP = 64_600

#: AASIST-L 아키텍처 설정 (vendor/aasist_config.conf)
MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor"


@dataclass(frozen=True)
class SpoofResult:
    """딥페이크 탐지 결과."""

    spoof_score: float
    """합성 음성일 확률 (0~1). 클수록 위조 가능성이 높다."""

    is_spoof: bool
    threshold: float
    segments_scored: int
    """점수를 낸 구간 수. 긴 오디오는 여러 구간으로 나눠 최댓값을 취한다."""

    @property
    def bonafide_score(self) -> float:
        return 1.0 - self.spoof_score


def _ensure_loaded(weights_path: str):
    """AASIST 모델을 지연 적재한다 (프로세스당 1회)."""
    global _model
    if _model is not None:
        return _model
    with _load_lock:
        if _model is not None:
            return _model

        # 모델 정의는 clovaai/aasist(MIT)에서 가져와 vendor/에 둔다. PyPI 패키지가
        # 없어 코드를 직접 가져오는 것 외에 방법이 없다.
        if str(VENDOR_DIR) not in sys.path:
            sys.path.insert(0, str(VENDOR_DIR))
        from aasist_model import Model  # type: ignore[import-not-found]

        logger.info("AASIST-L 모델 적재 중: %s", weights_path)
        model = Model(MODEL_CONFIG)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _model = model
        logger.info("AASIST-L 모델 적재 완료 (파라미터 %dK)", sum(p.numel() for p in model.parameters()) // 1000)
        return _model


def warmup(weights_path: str) -> None:
    _ensure_loaded(weights_path)


def is_loaded() -> bool:
    return _model is not None


def reset() -> None:
    """적재된 모델을 해제한다 (테스트용)."""
    global _model
    _model = None


def _segments(samples: np.ndarray, length: int = NB_SAMP) -> list[np.ndarray]:
    """오디오를 모델 입력 길이의 구간들로 나눈다.

    짧으면 반복해 채운다(원 저자 구현과 동일). 길면 겹치지 않는 구간으로 잘라
    각각 점수를 낸다 — 공격자가 긴 오디오의 일부에만 합성음을 섞는 경우를
    잡기 위해서다.
    """
    if len(samples) < length:
        repeats = int(np.ceil(length / max(len(samples), 1)))
        return [np.tile(samples, repeats)[:length]]

    count = len(samples) // length
    chunks = [samples[i * length : (i + 1) * length] for i in range(count)]
    remainder = len(samples) - count * length
    # 남는 꼬리가 절반 이상이면 끝에서 한 구간을 더 떼어 본다
    if remainder > length // 2:
        chunks.append(samples[-length:])
    return chunks


def detect(
    samples: np.ndarray,
    *,
    weights_path: str,
    threshold: float,
) -> SpoofResult:
    """합성 음성 여부를 판정한다.

    긴 오디오는 구간별로 점수를 내고 **최댓값**을 취한다. 일부만 합성이어도
    공격이므로, 평균을 내면 섞인 구간이 희석돼 놓친다.
    """
    model = _ensure_loaded(weights_path)
    chunks = _segments(samples)

    batch = torch.from_numpy(
        np.stack([np.ascontiguousarray(c, dtype=np.float32) for c in chunks])
    )
    with torch.no_grad():
        # 반환: (hidden, logits). logits[:, 0] = spoof, logits[:, 1] = bonafide
        _, logits = model(batch)
        probs = torch.softmax(logits, dim=1)

    spoof_scores = probs[:, 0].numpy()
    worst = float(spoof_scores.max())

    return SpoofResult(
        spoof_score=worst,
        is_spoof=worst >= threshold,
        threshold=threshold,
        segments_scored=len(chunks),
    )
