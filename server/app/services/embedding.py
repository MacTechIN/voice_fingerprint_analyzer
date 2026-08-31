"""ECAPA-TDNN 화자 임베딩 추출.

Phase 1 백본은 SpeechBrain의 `spkrec-ecapa-voxceleb` (192차원)이다.
Phase 6에서 WeSpeaker로 코어를 전환할 예정이므로 (02 §3.1), 호출부가 특정
구현에 묶이지 않도록 이 모듈이 유일한 접점이 되게 한다.

임베딩에는 반드시 모델 식별자와 차원을 함께 실어 보낸다. 모델을 교체하면
기존 벡터와 호환되지 않아 재등록이 필요한데, 메타데이터가 없으면 어떤 벡터가
어느 모델 것인지 사후에 알 수 없다 (02 §6 재등록 마이그레이션).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

_classifier = None
_load_lock = threading.Lock()


@dataclass(frozen=True)
class SpeakerEmbedding:
    """화자 임베딩과 그 출처 메타데이터."""

    vector: list[float]
    dim: int
    model: str
    l2_normalized: bool


def _ensure_loaded(model_name: str, cache_dir: str):
    """임베딩 모델을 지연 적재한다 (프로세스당 1회)."""
    global _classifier
    if _classifier is not None:
        return _classifier
    with _load_lock:
        if _classifier is not None:
            return _classifier
        from speechbrain.inference.speaker import EncoderClassifier

        logger.info("화자 임베딩 모델 적재 중: %s", model_name)
        _classifier = EncoderClassifier.from_hparams(
            source=model_name,
            savedir=f"{cache_dir}/{model_name.replace('/', '__')}",
            run_opts={"device": "cpu"},
        )
        logger.info("화자 임베딩 모델 적재 완료")
        return _classifier


def warmup(model_name: str, cache_dir: str) -> None:
    """모델을 미리 적재해 첫 요청의 콜드 스타트를 없앤다."""
    _ensure_loaded(model_name, cache_dir)


def extract(
    samples: np.ndarray,
    *,
    model_name: str,
    cache_dir: str,
    l2_normalize: bool = True,
) -> SpeakerEmbedding:
    """발화 파형에서 화자 임베딩을 추출한다.

    Args:
        samples: VAD를 통과한 16kHz mono float32 파형.
        l2_normalize: 코사인 유사도는 방향만 보므로 저장 단계에서 미리
            정규화해 두면 이후 내적만으로 유사도를 얻는다.
    """
    classifier = _ensure_loaded(model_name, cache_dir)

    wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        # 반환 shape: (batch, 1, dim)
        emb = classifier.encode_batch(wav).squeeze(0).squeeze(0)

    if l2_normalize:
        norm = torch.linalg.vector_norm(emb)
        # 정규화된 발화가 완전 무음이면 norm이 0이 될 수 있다. VAD가 앞단에서
        # 걸러주지만, 0으로 나누어 NaN을 DB에 저장하는 사고를 막는다.
        if norm > 0:
            emb = emb / norm

    vector = emb.tolist()
    return SpeakerEmbedding(
        vector=vector,
        dim=len(vector),
        model=model_name,
        l2_normalized=l2_normalize,
    )
