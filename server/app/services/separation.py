"""다중 화자 음성 분리 및 타겟 화자 선택 (Phase 7).

혼합 음성에서 개별 화자를 분리한 뒤, **등록 성문과 가장 잘 맞는 출력을 골라**
검증에 넘긴다. 이렇게 하면 분리망의 출력 순서가 매번 달라지는 순열 문제
(Permutation Problem, 02 §2.3)를 실용적으로 회피할 수 있다.

## 왜 진짜 TSE가 아니라 "분리 + 선택"인가

02 §2.3은 등록 임베딩을 분리망에 **조건으로 주입**하는 타겟 화자 추출(TSE)을
권장했다. 구현체 후보였던 `wenet-e2e/wesep`은 저장소에 **라이선스 파일이 없어**
채택 전제 조건을 만족하지 못한다 (08 §3).

대안으로 택한 "모든 화자를 분리한 뒤 등록 임베딩과 대조해 고르기"는 조건부 분리가
아니므로 비타겟 화자를 능동적으로 억압하지는 못한다. 그러나 **순열 문제를 없앤다는
실용적 목표는 동일하게 달성**하며, 라이선스가 명확한 Apache-2.0 모델만 쓴다.
wesep 라이선스가 확인되면 진짜 TSE로 교체할 수 있도록 이 모듈의 인터페이스는
"혼합 + 등록 임베딩 → 타겟 파형" 형태로 두었다.

## 분리 모델 선택

06 Phase 7은 ClearerVoice-Studio(MossFormer2-SS)를 1순위로 적었으나, 실제 설치를
검토한 결과 `clearvoice` 패키지가 opencv·torchvision·scenedetect 등 영상 처리
의존성을 끌어오고 **soundfile을 다운그레이드**한다. 파이프라인이 의존하는 패키지라
위험이 이득보다 크다고 판단해, 이미 검증된 SpeechBrain SepFormer를 쓴다.
`sepformer-whamr16k`는 16kHz라 리샘플링 왕복이 없고 잡음·잔향 조건으로 학습됐다.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

_model = None
_model_key: str | None = None
_load_lock = threading.Lock()

#: 16kHz 2화자 분리. 8kHz 모델(wsj02mix)을 쓰면 16k→8k→16k 왕복에서 4~8kHz
#: 대역이 통째로 날아가는데, 그 대역은 화자 식별에 기여한다.
DEFAULT_MODEL = "speechbrain/sepformer-whamr16k"


@dataclass(frozen=True)
class SeparatedSource:
    """분리된 화자 하나."""

    index: int
    samples: np.ndarray
    similarity: float
    """등록 임베딩과의 코사인 유사도. 타겟 선택의 근거다."""


@dataclass(frozen=True)
class SeparationResult:
    """분리 및 타겟 선택 결과."""

    target: np.ndarray
    """등록 성문과 가장 잘 맞는 출력."""

    target_index: int
    target_similarity: float
    runner_up_similarity: float | None
    """두 번째로 잘 맞는 출력의 유사도."""

    source_count: int

    @property
    def selection_margin(self) -> float | None:
        """1등과 2등의 유사도 차이.

        작으면 어느 출력이 타겟인지 모호하다는 뜻이다 — 분리가 제대로 안 됐거나
        두 화자의 목소리가 실제로 비슷한 경우이며, 어느 쪽이든 판정을 신뢰하기
        어렵다는 신호다.
        """
        if self.runner_up_similarity is None:
            return None
        return self.target_similarity - self.runner_up_similarity


def _ensure_loaded(model_name: str, cache_dir: str):
    """분리 모델을 지연 적재한다 (프로세스당 1회)."""
    global _model, _model_key
    if _model is not None and _model_key == model_name:
        return _model
    with _load_lock:
        if _model is not None and _model_key == model_name:
            return _model
        from speechbrain.inference.separation import SepformerSeparation

        logger.info("음성 분리 모델 적재 중: %s", model_name)
        _model = SepformerSeparation.from_hparams(
            source=model_name,
            savedir=f"{cache_dir}/{model_name.replace('/', '__')}",
            run_opts={"device": "cpu"},
        )
        _model_key = model_name
        logger.info("음성 분리 모델 적재 완료")
        return _model


def warmup(model_name: str = DEFAULT_MODEL, cache_dir: str = "./.model_cache") -> None:
    _ensure_loaded(model_name, cache_dir)


def is_loaded() -> bool:
    return _model is not None


def reset() -> None:
    """적재된 모델을 해제한다 (테스트용)."""
    global _model, _model_key
    _model = None
    _model_key = None


def separate(
    samples: np.ndarray,
    *,
    model_name: str = DEFAULT_MODEL,
    cache_dir: str = "./.model_cache",
) -> list[np.ndarray]:
    """혼합 파형을 화자별 파형으로 분리한다.

    Returns:
        분리된 소스 목록. 순서는 **의미가 없다** — 매 호출마다 달라질 수 있는
        것이 순열 문제의 본질이다.
    """
    model = _ensure_loaded(model_name, cache_dir)

    wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        # 반환 shape: (batch, time, n_src)
        estimates = model.separate_batch(wav)

    sources = estimates.squeeze(0).numpy()
    return [np.ascontiguousarray(sources[:, i], dtype=np.float32) for i in range(sources.shape[1])]


def extract_target(
    samples: np.ndarray,
    reference_embedding: list[float] | np.ndarray,
    *,
    embed_fn,
    model_name: str = DEFAULT_MODEL,
    cache_dir: str = "./.model_cache",
) -> SeparationResult:
    """혼합에서 등록 화자에 해당하는 파형을 골라낸다.

    Args:
        samples: 혼합 파형 (16kHz mono).
        reference_embedding: 등록된 타겟 화자의 성문.
        embed_fn: 파형 → 임베딩(list[float]) 함수. 임베딩 백엔드에 의존하지
            않도록 주입받는다.

    분리 출력이 하나뿐이면 그대로 반환한다. 분리망이 항상 고정 개수(보통 2)를
    내므로 실제로는 드물지만, 모델을 바꿔도 깨지지 않게 처리한다.
    """
    sources = separate(samples, model_name=model_name, cache_dir=cache_dir)

    ref = np.asarray(reference_embedding, dtype=np.float32)
    ref_norm = np.linalg.norm(ref)
    if ref_norm > 0:
        ref = ref / ref_norm

    scored: list[SeparatedSource] = []
    for i, source in enumerate(sources):
        vec = np.asarray(embed_fn(source), dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        scored.append(
            SeparatedSource(index=i, samples=source, similarity=float(ref @ vec))
        )

    scored.sort(key=lambda s: -s.similarity)
    best = scored[0]
    runner_up = scored[1].similarity if len(scored) > 1 else None

    logger.info(
        "타겟 화자 선택: 출력%d (유사도 %.4f, 차순위 %s)",
        best.index,
        best.similarity,
        f"{runner_up:.4f}" if runner_up is not None else "-",
    )
    return SeparationResult(
        target=best.samples,
        target_index=best.index,
        target_similarity=best.similarity,
        runner_up_similarity=runner_up,
        source_count=len(sources),
    )
