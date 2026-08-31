"""AS-Norm 정규화 단위 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from app.services import asnorm, scoring


def _unit(vec: list[float]) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_normalize_matches_formula():
    """02 §4.2의 대칭 공식과 정확히 일치하는지 손계산으로 확인한다.

        S = ½ · [ (raw − μ_e)/σ_e + (raw − μ_t)/σ_t ]
    """
    enroll_sims = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    test_sims = np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float32)
    raw = 0.8

    # top_k=4면 전체를 쓴다
    got = asnorm.normalize(raw, enroll_sims, test_sims, top_k=4)

    e_mean, e_std = enroll_sims.mean(), enroll_sims.std()
    t_mean, t_std = test_sims.mean(), test_sims.std()
    expected = 0.5 * ((raw - e_mean) / e_std + (raw - t_mean) / t_std)

    assert got == pytest.approx(expected, rel=1e-5)


def test_normalize_uses_only_top_k():
    """상위 K개만 사용한다 — '적응형'의 핵심.

    낮은 유사도를 잔뜩 붙여도 상위 K개가 그대로면 결과가 변하지 않아야 한다.
    """
    top = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    padded = np.concatenate([np.full(50, -0.9, dtype=np.float32), top])

    a = asnorm.normalize(0.8, top, top, top_k=3)
    b = asnorm.normalize(0.8, padded, padded, top_k=3)

    assert a == pytest.approx(b, rel=1e-5)


def test_normalize_is_symmetric():
    """등록·검증을 맞바꿔도 같은 점수 — 대칭성이 AS-Norm의 정의다."""
    e = np.array([0.1, 0.3, 0.5], dtype=np.float32)
    t = np.array([0.2, 0.25, 0.9], dtype=np.float32)

    forward = asnorm.normalize(0.7, e, t, top_k=3)
    backward = asnorm.normalize(0.7, t, e, top_k=3)

    assert forward == pytest.approx(backward, rel=1e-6)


def test_normalize_survives_zero_variance_cohort():
    """코호트 유사도가 전부 같아도 0으로 나누지 않는다."""
    flat = np.full(10, 0.3, dtype=np.float32)
    result = asnorm.normalize(0.9, flat, flat, top_k=5)
    assert np.isfinite(result)


def test_normalize_rejects_empty_cohort():
    with pytest.raises(ValueError, match="코호트가 비어 있습니다"):
        asnorm.normalize(0.5, np.array([]), np.array([]), top_k=10)


def test_cohort_index_computes_cosine_similarity():
    """CohortIndex가 정규화 여부와 무관하게 코사인 유사도를 낸다."""
    # 크기를 일부러 다르게 준다 — 내적이 아니라 코사인이어야 한다
    cohort = np.array([[3.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    index = asnorm.CohortIndex(cohort, model="m", top_k=2)

    sims = index.similarities([1.0, 0.0])

    assert sims[0] == pytest.approx(1.0, abs=1e-6)
    assert sims[1] == pytest.approx(0.0, abs=1e-6)
    assert index.size == 2
    assert index.dim == 2


def test_cohort_index_rejects_non_matrix():
    with pytest.raises(ValueError, match="행렬"):
        asnorm.CohortIndex(np.array([1.0, 2.0], dtype=np.float32), model="m", top_k=1)


def test_asnorm_separates_better_than_raw_cosine():
    """AS-Norm이 실제로 genuine/impostor 분리를 개선하는지 확인한다.

    등록 화자와 가까운 코호트를 두어, 원시 코사인만으로는 genuine과 impostor가
    겹치지만 코호트 기준으로 표준화하면 벌어지는 상황을 만든다.
    """
    rng = np.random.default_rng(7)
    dim = 32

    speaker = _unit(rng.normal(size=dim).tolist())
    # 같은 화자의 다른 발화 — 약간 흔들린 벡터
    genuine = _unit((speaker + 0.35 * rng.normal(size=dim)).tolist())
    impostor = _unit(rng.normal(size=dim).tolist())

    cohort = np.stack([_unit(rng.normal(size=dim).tolist()) for _ in range(200)])
    index = asnorm.CohortIndex(cohort, model="m", top_k=50)

    raw_genuine = scoring.cosine_similarity(speaker.tolist(), genuine.tolist())
    raw_impostor = scoring.cosine_similarity(speaker.tolist(), impostor.tolist())

    norm_genuine = index.normalize(raw_genuine, speaker, genuine)
    norm_impostor = index.normalize(raw_impostor, speaker, impostor)

    # 정규화 후에도 genuine이 impostor보다 높아야 한다 (순서 보존)
    assert norm_genuine > norm_impostor
    # 정규화 점수는 코호트 표준편차 단위이므로 코사인의 [-1,1] 범위를 벗어난다
    assert abs(norm_genuine) > 1.0


def test_match_normalized_reports_both_scores():
    """정규화 판정 결과가 원시 점수와 정규화 점수를 모두 담는다."""
    rng = np.random.default_rng(3)
    dim = 16
    v = _unit(rng.normal(size=dim).tolist())
    cohort = np.stack([_unit(rng.normal(size=dim).tolist()) for _ in range(50)])
    index = asnorm.CohortIndex(cohort, model="m", top_k=20)

    result = scoring.match_normalized(v.tolist(), v.tolist(), cohort=index, threshold=0.0)

    assert result.raw_cosine == pytest.approx(1.0, abs=1e-5)
    assert result.normalized_score is not None
    assert result.scoring_method == "as_norm"
    assert result.decision_score == result.normalized_score
    assert result.is_verified is True


def test_percentage_normalized_anchors_at_threshold():
    """AS-Norm 퍼센트도 임계값에서 50%다."""
    assert scoring.to_percentage_normalized(2.5, threshold=2.5) == pytest.approx(50.0)
    assert scoring.to_percentage_normalized(100.0, threshold=2.5) == 100.0
    assert scoring.to_percentage_normalized(-100.0, threshold=2.5) == 0.0


def test_match_best_normalized_picks_highest_normalized_score():
    """여러 등록 중 정규화 점수가 가장 높은 것으로 판정한다."""
    rng = np.random.default_rng(11)
    dim = 16
    probe = _unit(rng.normal(size=dim).tolist())
    far = _unit(rng.normal(size=dim).tolist())
    cohort = np.stack([_unit(rng.normal(size=dim).tolist()) for _ in range(60)])
    index = asnorm.CohortIndex(cohort, model="m", top_k=20)

    result = scoring.match_best_normalized(
        probe.tolist(), [far.tolist(), probe.tolist()], cohort=index, threshold=0.0
    )

    assert result.raw_cosine == pytest.approx(1.0, abs=1e-5)
