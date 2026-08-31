"""화자 검증 성능 지표.

EER과 minDCF는 임계값 결정의 근거다. 임계값을 관례값으로 두면 시스템이 실제로
어느 지점에서 실패하는지 알 수 없다 (06 Phase 6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetPoint:
    """한 임계값에서의 오류율."""

    threshold: float
    far: float
    """오수락률 (False Acceptance Rate) — 타인을 본인으로 잘못 승인."""
    frr: float
    """오거부률 (False Rejection Rate) — 본인을 타인으로 잘못 거부."""


@dataclass(frozen=True)
class Metrics:
    """평가 결과 요약."""

    eer: float
    """FAR과 FRR이 같아지는 지점의 오류율."""
    eer_threshold: float
    """EER 지점의 임계값 — 두 오류를 대칭적으로 다룰 때의 운영 임계값 후보."""
    min_dcf: float
    """최소 탐지 비용 (p_target=0.01, C_miss=C_fa=1)."""
    min_dcf_threshold: float
    genuine_mean: float
    genuine_std: float
    impostor_mean: float
    impostor_std: float
    n_genuine: int
    n_impostor: int

    @property
    def separation(self) -> float:
        """두 분포의 평균 간격을 표준편차로 나눈 값 (d-prime 유사).

        클수록 분리가 좋다. EER과 함께 보면 성능 변화의 원인을 읽기 쉽다.
        """
        pooled = np.sqrt((self.genuine_std**2 + self.impostor_std**2) / 2)
        if pooled == 0:
            return 0.0
        return float((self.genuine_mean - self.impostor_mean) / pooled)


def det_curve(scores: np.ndarray, labels: np.ndarray) -> list[DetPoint]:
    """모든 후보 임계값에서 FAR/FRR을 계산한다.

    Args:
        scores: 유사도 점수 (높을수록 동일인).
        labels: 1이면 동일 화자(genuine), 0이면 타인(impostor).
    """
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    n_genuine = int(labels.sum())
    n_impostor = int(len(labels) - n_genuine)
    if n_genuine == 0 or n_impostor == 0:
        raise ValueError("genuine과 impostor 트라이얼이 모두 필요합니다")

    # 임계값을 낮은 점수부터 올리며 누적한다.
    # threshold=t 에서 승인 조건은 score >= t.
    genuine_below = np.cumsum(sorted_labels)          # 거부된 genuine 수 = FRR 분자
    impostor_below = np.cumsum(1 - sorted_labels)     # 거부된 impostor 수

    points: list[DetPoint] = []
    for i, thr in enumerate(sorted_scores):
        frr = genuine_below[i - 1] / n_genuine if i > 0 else 0.0
        far = (n_impostor - (impostor_below[i - 1] if i > 0 else 0)) / n_impostor
        points.append(DetPoint(threshold=float(thr), far=float(far), frr=float(frr)))
    return points


def compute(scores: np.ndarray, labels: np.ndarray, *, p_target: float = 0.01) -> Metrics:
    """EER, minDCF, 점수 분포 통계를 계산한다."""
    points = det_curve(scores, labels)

    # EER: |FAR - FRR|이 최소인 지점
    eer_point = min(points, key=lambda p: abs(p.far - p.frr))
    eer = (eer_point.far + eer_point.frr) / 2

    # minDCF: C_miss=C_fa=1 인 정규화 탐지 비용의 최소값
    def dcf(p: DetPoint) -> float:
        cost = p_target * p.frr + (1 - p_target) * p.far
        return cost / min(p_target, 1 - p_target)

    dcf_point = min(points, key=dcf)

    genuine = scores[labels == 1]
    impostor = scores[labels == 0]

    return Metrics(
        eer=float(eer),
        eer_threshold=float(eer_point.threshold),
        min_dcf=float(dcf(dcf_point)),
        min_dcf_threshold=float(dcf_point.threshold),
        genuine_mean=float(genuine.mean()),
        genuine_std=float(genuine.std()),
        impostor_mean=float(impostor.mean()),
        impostor_std=float(impostor.std()),
        n_genuine=int(len(genuine)),
        n_impostor=int(len(impostor)),
    )


def error_rates_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    """특정 임계값에서의 (FAR, FRR)."""
    accepted = scores >= threshold
    genuine = labels == 1
    impostor = labels == 0
    far = float((accepted & impostor).sum() / max(impostor.sum(), 1))
    frr = float((~accepted & genuine).sum() / max(genuine.sum(), 1))
    return far, frr
