"""성문 유사도 산출 및 판정.

Phase 2는 **원시 코사인 유사도 + 고정 임계값** 판정이다. 02 §4.2가 지적하듯
이 방식은 화자 상태·발화 길이·잔존 소음에 따라 점수 편차가 커서 실무 환경의
실패율이 높다. Phase 6에서 AS-Norm 정규화를 필수 경로로 넣기 전까지는
**프로토타입 수준의 판정**임을 전제로 사용해야 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MatchResult:
    """1:1 대조 결과."""

    raw_cosine: float
    match_probability: float
    is_verified: bool
    threshold: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """두 임베딩의 코사인 유사도.

    저장된 벡터는 이미 L2 정규화되어 있으므로 통상 내적만으로 충분하지만,
    정규화되지 않은 벡터가 섞여 들어와도 올바른 값이 나오도록 노름으로 나눈다.
    """
    if len(a) != len(b):
        raise ValueError(f"임베딩 차원이 다릅니다: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    # 부동소수 오차로 |cos|가 1을 아주 살짝 넘을 수 있다
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def to_percentage(cosine: float, threshold: float) -> float:
    """코사인 유사도를 0~100 척도로 변환한다 (FR-06).

    **이 값은 확률이 아니다.** 판정 경계를 기준으로 한 구간 선형 재척도이며,
    `cosine == threshold`가 정확히 50%가 되도록 맞춰 사용자가 "경계에 얼마나
    가까운지"를 직관적으로 읽게 한 것이다.

    진짜 확률로 쓰려면 Genuine/Impostor 점수 분포를 모아 캘리브레이션해야 하며,
    그것은 AS-Norm과 EER 측정이 들어오는 Phase 6 과제다 (06 Phase 6).
    """
    if cosine >= threshold:
        span = 1.0 - threshold
        ratio = 1.0 if span <= 0 else (cosine - threshold) / span
        return 50.0 + 50.0 * ratio
    span = threshold - (-1.0)
    ratio = 0.0 if span <= 0 else (cosine - (-1.0)) / span
    return 50.0 * ratio


def match(probe: list[float], reference: list[float], threshold: float) -> MatchResult:
    """검증 임베딩을 등록 임베딩과 1:1 대조한다."""
    cosine = cosine_similarity(probe, reference)
    return MatchResult(
        raw_cosine=cosine,
        match_probability=to_percentage(cosine, threshold),
        is_verified=cosine >= threshold,
        threshold=threshold,
    )


def match_best(probe: list[float], references: list[list[float]], threshold: float) -> MatchResult:
    """여러 등록 발화 중 가장 잘 맞는 것으로 판정한다.

    한 사용자가 여러 번 등록했을 때, 평균 벡터를 쓰는 대신 최대 유사도를 택한다.
    평균은 발화 조건이 다른 벡터들을 뭉개 어느 쪽과도 덜 닮은 중심을 만들 수
    있다. 다중 등록 발화의 평균 성문은 01 §2의 향후 보강 후보로 남겨둔다.
    """
    if not references:
        raise ValueError("등록된 성문이 없습니다")
    return max(
        (match(probe, ref, threshold) for ref in references),
        key=lambda r: r.raw_cosine,
    )
