"""적응형 대칭 점수 정규화 (AS-Norm).

원시 코사인에 고정 임계값을 적용하면 화자 상태·발화 길이·잔존 소음에 따라
점수 편차가 커서 실무 실패율이 높다 (02 §4.2). AS-Norm은 임포스터 코호트를
기준으로 점수를 표준화해 이 편차를 흡수한다.

    S_AS-Norm = ½ · [ (S(e,t) − μ_c(e)) / σ_c(e) + (S(e,t) − μ_c(t)) / σ_c(t) ]

μ_c(x), σ_c(x)는 x와 코호트 간 유사도 중 **상위 K개**의 평균·표준편차다.
상위 K개만 쓰는 것이 '적응형(Adaptive)'의 의미로, 가장 혼동하기 쉬운 임포스터를
기준으로 삼아 판정 경계를 엄격하게 만든다.

코호트는 실제 사용자와 무관한 화자여야 한다. 사용자 성문이 섞이면 사칭자 통계가
오염되어 정규화가 자기 자신을 참조하게 된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CohortStats:
    """한 임베딩에 대한 코호트 통계."""

    mean: float
    std: float
    top_k: int


def _cohort_stats(similarities: np.ndarray, top_k: int, *, eps: float = 1e-6) -> CohortStats:
    """코호트 유사도 상위 K개의 평균·표준편차."""
    k = min(top_k, len(similarities))
    if k <= 0:
        raise ValueError("코호트가 비어 있습니다")

    # 상위 k개만 남긴다 (전체 정렬 없이 부분 선택)
    top = np.partition(similarities, -k)[-k:] if k < len(similarities) else similarities
    std = float(top.std())
    # 코호트가 1개거나 값이 모두 같으면 표준편차가 0이 되어 0으로 나누게 된다.
    # eps로 막되, 이 경우 정규화는 사실상 평균 차감만 수행한다.
    return CohortStats(mean=float(top.mean()), std=max(std, eps), top_k=k)


def normalize(
    raw_score: float,
    enroll_vs_cohort: np.ndarray,
    test_vs_cohort: np.ndarray,
    *,
    top_k: int,
) -> float:
    """AS-Norm 대칭 정규화 점수를 계산한다.

    Args:
        raw_score: 등록·검증 임베딩 간 원시 코사인 유사도.
        enroll_vs_cohort: 등록 임베딩과 코호트 전체의 유사도.
        test_vs_cohort: 검증 임베딩과 코호트 전체의 유사도.
        top_k: 적응적으로 선택할 상위 코호트 수.
    """
    e = _cohort_stats(enroll_vs_cohort, top_k)
    t = _cohort_stats(test_vs_cohort, top_k)
    return 0.5 * ((raw_score - e.mean) / e.std + (raw_score - t.mean) / t.std)


class CohortIndex:
    """메모리에 적재된 임포스터 코호트.

    코호트는 등록 성문 수와 무관하게 고정 크기이고 요청마다 전부 참조되므로,
    매 검증마다 DB에서 읽는 대신 기동 시 한 번 적재해 행렬 곱으로 처리한다.
    """

    def __init__(self, embeddings: np.ndarray, model: str, top_k: int) -> None:
        if embeddings.ndim != 2:
            raise ValueError("코호트는 (N, dim) 행렬이어야 합니다")
        # 코사인 유사도를 내적으로 얻기 위해 미리 정규화한다
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = (embeddings / norms).astype(np.float32)
        self.model = model
        self.top_k = top_k

    @property
    def size(self) -> int:
        return self._matrix.shape[0]

    @property
    def dim(self) -> int:
        return self._matrix.shape[1]

    def similarities(self, embedding: list[float] | np.ndarray) -> np.ndarray:
        """임베딩과 코호트 전체의 코사인 유사도."""
        v = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return self._matrix @ v

    def normalize(
        self, raw_score: float, enroll: list[float] | np.ndarray, test: list[float] | np.ndarray
    ) -> float:
        """등록·검증 임베딩 쌍의 원시 점수를 정규화한다."""
        return normalize(
            raw_score,
            self.similarities(enroll),
            self.similarities(test),
            top_k=self.top_k,
        )
