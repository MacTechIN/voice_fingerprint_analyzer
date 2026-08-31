"""임베딩 단위 테스트 — 실제 ECAPA-TDNN 모델 사용."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import get_settings
from app.services import embedding as embedding_svc

from .conftest import synth_speech

pytestmark = pytest.mark.usefixtures("client")  # 모델 적재를 세션 픽스처에 위임


def _extract(samples):
    s = get_settings()
    return embedding_svc.extract(
        samples, model_name=s.embedding_model, cache_dir=s.model_cache_dir
    )


def test_embedding_shape_and_metadata():
    """ECAPA-TDNN은 192차원 벡터를 만들고, 출처 메타데이터가 함께 온다."""
    emb = _extract(synth_speech(3.0))

    assert emb.dim == 192
    assert len(emb.vector) == 192
    assert emb.model == get_settings().embedding_model
    assert emb.l2_normalized is True
    assert all(np.isfinite(emb.vector))


def test_embedding_is_l2_normalized():
    """정규화된 벡터의 노름은 1이다 — 이후 내적만으로 코사인 유사도를 얻는다."""
    emb = _extract(synth_speech(3.0))
    assert np.linalg.norm(emb.vector) == pytest.approx(1.0, abs=1e-5)


def test_embedding_is_deterministic():
    """같은 입력은 같은 벡터를 낸다 — 재현성이 없으면 검증 결과를 신뢰할 수 없다."""
    samples = synth_speech(3.0)
    first = _extract(samples)
    second = _extract(samples)

    assert np.allclose(first.vector, second.vector, atol=1e-6)


def test_embedding_discriminates_different_sources():
    """음향적으로 다른 입력은 다른 방향의 벡터가 된다.

    합성음이므로 화자 인식 성능 자체를 재는 것은 아니다. 모델이 입력에 반응해
    서로 다른 임베딩을 낸다는 것 — 즉 상수를 뱉지 않는다는 것 — 만 확인한다.
    """
    low = _extract(synth_speech(3.0, f0=95.0, seed=1))
    high = _extract(synth_speech(3.0, f0=210.0, seed=2))

    cosine = float(np.dot(low.vector, high.vector))
    assert cosine < 0.99
