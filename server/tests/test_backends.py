"""임베딩 백엔드 교체 및 음성 향상 테스트.

WeSpeaker ONNX와 DeepFilterNet은 모델을 내려받아야 하므로, 캐시가 없는 환경에서는
건너뛴다. `VG_TEST_HEAVY=1`을 주면 강제로 실행한다.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.config import get_settings
from app.services import embedding as embedding_svc

from .conftest import synth_speech

WESPEAKER_MODEL = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
SPEECHBRAIN_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
HEAVY = os.environ.get("VG_TEST_HEAVY") == "1"


def _cache_dir() -> str:
    return get_settings().model_cache_dir


def _wespeaker_cached() -> bool:
    """모델이 이미 캐시되어 있는지 — 네트워크 없이 돌 수 있는지 판단."""
    from pathlib import Path

    root = Path(_cache_dir()) / "hf"
    if not root.exists():
        return False
    return any(root.rglob("*resnet34_LM.onnx"))


requires_wespeaker = pytest.mark.skipif(
    not (HEAVY or _wespeaker_cached()),
    reason="WeSpeaker ONNX 모델 미캐시 (VG_TEST_HEAVY=1로 강제 실행)",
)


class TestBackendSelection:
    """백엔드 선택 로직 — 모델 다운로드가 필요 없는 부분."""

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="알 수 없는 임베딩 백엔드"):
            embedding_svc._make_backend("nonexistent", "some/model", _cache_dir())

    def test_unknown_wespeaker_model_is_rejected(self):
        """지원 목록에 없는 모델은 파일명을 추측하지 않고 거부한다."""
        with pytest.raises(ValueError, match="알 수 없는 WeSpeaker 모델"):
            embedding_svc._WeSpeakerOnnxBackend("Wespeaker/not-a-real-model", _cache_dir())

    def test_known_models_map_to_onnx_files(self):
        """지원 목록의 파일명이 모두 .onnx인지 — 오타 방지."""
        for model, filename in embedding_svc._WeSpeakerOnnxBackend.KNOWN_MODELS.items():
            assert model.startswith("Wespeaker/")
            assert filename.endswith(".onnx")


@requires_wespeaker
class TestWeSpeakerBackend:
    """WeSpeaker ONNX 백엔드 실동작."""

    def test_produces_256_dim_embedding(self):
        """ResNet34-LM은 256차원 — SpeechBrain의 192와 다르다."""
        emb = embedding_svc.extract(
            synth_speech(3.0),
            model_name=WESPEAKER_MODEL,
            cache_dir=_cache_dir(),
            backend="wespeaker",
        )

        assert emb.dim == 256
        assert emb.model == WESPEAKER_MODEL
        assert emb.l2_normalized is True
        assert np.linalg.norm(emb.vector) == pytest.approx(1.0, abs=1e-5)
        assert all(np.isfinite(emb.vector))

    def test_is_deterministic(self):
        """dither를 껐으므로 같은 입력은 같은 벡터를 낸다."""
        samples = synth_speech(3.0)
        kwargs = dict(
            model_name=WESPEAKER_MODEL, cache_dir=_cache_dir(), backend="wespeaker"
        )
        first = embedding_svc.extract(samples, **kwargs)
        second = embedding_svc.extract(samples, **kwargs)

        assert np.allclose(first.vector, second.vector, atol=1e-6)

    def test_discriminates_different_sources(self):
        kwargs = dict(
            model_name=WESPEAKER_MODEL, cache_dir=_cache_dir(), backend="wespeaker"
        )
        low = embedding_svc.extract(synth_speech(3.0, f0=95.0, seed=1), **kwargs)
        high = embedding_svc.extract(synth_speech(3.0, f0=210.0, seed=2), **kwargs)

        assert float(np.dot(low.vector, high.vector)) < 0.99

    def test_switching_backend_reloads_model(self):
        """백엔드를 바꾸면 새 모델이 적재되어 차원이 달라진다.

        캐시 키가 백엔드를 포함하지 않으면 이전 모델을 계속 쓰게 되는데, 오류가
        나지 않아 눈치채기 어렵다.
        """
        embedding_svc.reset()
        try:
            sb = embedding_svc.extract(
                synth_speech(3.0),
                model_name=SPEECHBRAIN_MODEL,
                cache_dir=_cache_dir(),
                backend="speechbrain",
            )
            ws = embedding_svc.extract(
                synth_speech(3.0),
                model_name=WESPEAKER_MODEL,
                cache_dir=_cache_dir(),
                backend="wespeaker",
            )
            assert sb.dim == 192
            assert ws.dim == 256
        finally:
            embedding_svc.reset()


class TestNoiseInjection:
    """소음 주입 유틸 — 향상 평가의 전제."""

    @staticmethod
    def _measure_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
        """스케일 불변 SNR.

        `add_noise`는 클리핑을 피하려 혼합물 전체에 균일 게인을 걸 수 있다.
        그래서 `noisy - clean`은 순수 잡음이 아니다. 먼저 noisy를 clean에
        투영해 게인을 되찾은 뒤 잔차를 잡음으로 본다 — SI-SNR과 같은 방식이다.
        """
        alpha = float(np.dot(noisy, clean) / np.dot(clean, clean))
        signal = alpha * clean
        residual = noisy - signal
        return float(10 * np.log10(np.sum(signal**2) / np.sum(residual**2)))

    def test_add_noise_hits_target_snr(self):
        """요청한 SNR에 실제로 맞는지 확인한다.

        SNR이 틀리면 향상 효과 측정 전체가 무의미해진다.
        """
        from eval.noise_eval import add_noise

        clean = synth_speech(2.0)
        rng = np.random.default_rng(0)

        for target_snr in (20.0, 10.0, 5.0, 0.0):
            noisy = add_noise(clean, target_snr, rng)
            measured = self._measure_snr(clean, noisy)
            assert measured == pytest.approx(target_snr, abs=0.5)

    def test_add_noise_avoids_clipping(self):
        from eval.noise_eval import add_noise

        clean = synth_speech(2.0)
        noisy = add_noise(clean, 0.0, np.random.default_rng(0))

        assert np.abs(noisy).max() <= 1.0

    def test_lower_snr_is_noisier(self):
        """SNR이 낮을수록 잡음 성분이 커야 한다 — 부호 실수를 잡는다."""
        from eval.noise_eval import add_noise

        clean = synth_speech(2.0)
        rng = np.random.default_rng(0)
        quiet = self._measure_snr(clean, add_noise(clean, 20.0, rng))
        loud = self._measure_snr(clean, add_noise(clean, 0.0, rng))

        assert loud < quiet
