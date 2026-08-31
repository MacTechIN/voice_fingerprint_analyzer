"""스코어링 단위 테스트."""

from __future__ import annotations

import math

import pytest

from app.services import scoring


def test_cosine_of_identical_vectors_is_one():
    v = [0.6, 0.8, 0.0]
    assert scoring.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert scoring.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_of_opposite_vectors_is_minus_one():
    assert scoring.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_ignores_magnitude():
    """정규화되지 않은 벡터가 섞여도 방향만 본다."""
    assert scoring.cosine_similarity([1.0, 0.0], [7.0, 0.0]) == pytest.approx(1.0)


def test_cosine_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="차원"):
        scoring.cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_cosine_handles_zero_vector():
    """영벡터는 방향이 없으므로 0을 반환한다 — NaN을 흘리지 않는다."""
    assert scoring.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_percentage_is_fifty_at_threshold():
    """임계값에서 정확히 50% — 사용자가 경계와의 거리를 읽을 수 있게 한 기준점."""
    assert scoring.to_percentage(0.25, threshold=0.25) == pytest.approx(50.0)


def test_percentage_endpoints():
    assert scoring.to_percentage(1.0, threshold=0.25) == pytest.approx(100.0)
    assert scoring.to_percentage(-1.0, threshold=0.25) == pytest.approx(0.0)


def test_percentage_is_monotonic():
    """유사도가 오르면 퍼센트도 오른다 — 뒤집히면 UI가 거짓말을 한다."""
    values = [scoring.to_percentage(c / 10, threshold=0.25) for c in range(-10, 11)]
    assert values == sorted(values)
    assert all(0.0 <= v <= 100.0 for v in values)


def test_match_verifies_above_threshold():
    v = [1.0, 0.0]
    result = scoring.match(v, v, threshold=0.25)

    assert result.is_verified is True
    assert result.raw_cosine == pytest.approx(1.0)
    assert result.match_probability == pytest.approx(100.0)
    assert result.threshold == 0.25


def test_match_rejects_below_threshold():
    result = scoring.match([1.0, 0.0], [0.0, 1.0], threshold=0.25)

    assert result.is_verified is False
    assert result.raw_cosine == pytest.approx(0.0)
    assert result.match_probability < 50.0


def test_match_best_picks_highest_similarity():
    """여러 등록 발화 중 가장 잘 맞는 것으로 판정한다."""
    probe = [1.0, 0.0]
    refs = [[0.0, 1.0], [0.9, math.sqrt(1 - 0.81)], [-1.0, 0.0]]

    result = scoring.match_best(probe, refs, threshold=0.25)

    assert result.raw_cosine == pytest.approx(0.9, abs=1e-6)
    assert result.is_verified is True


def test_match_best_requires_references():
    with pytest.raises(ValueError, match="등록된 성문이 없습니다"):
        scoring.match_best([1.0, 0.0], [], threshold=0.25)
