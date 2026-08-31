"""음성 분리 및 타겟 선택 테스트.

분리 모델은 무겁게 내려받으므로, 캐시가 없으면 모델을 쓰는 테스트는 건너뛴다.
`VG_TEST_HEAVY=1`로 강제 실행할 수 있다.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from app.services import separation as separation_svc

from .conftest import SAMPLE_RATE, synth_speech

HEAVY = os.environ.get("VG_TEST_HEAVY") == "1"


def _model_cached() -> bool:
    root = Path("./.model_cache")
    return root.exists() and any(root.glob("*sepformer*"))


requires_model = pytest.mark.skipif(
    not (HEAVY or _model_cached()),
    reason="SepFormer 모델 미캐시 (VG_TEST_HEAVY=1로 강제 실행)",
)


def _unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


class TestTargetSelection:
    """타겟 선택 로직 — 분리 모델 없이 검증할 수 있는 부분.

    `extract_target`의 선택 규칙은 분리 결과와 무관하게 성립해야 한다. 분리를
    가짜로 대체해 규칙만 검사한다.
    """

    @pytest.fixture(autouse=True)
    def fake_separate(self, monkeypatch):
        """분리를 두 개의 고정 파형으로 대체한다."""
        self.source_a = synth_speech(2.0, f0=110.0, seed=1)
        self.source_b = synth_speech(2.0, f0=230.0, seed=2)
        monkeypatch.setattr(
            separation_svc,
            "separate",
            lambda samples, **kw: [self.source_a, self.source_b],
        )

    def test_picks_source_matching_reference(self):
        """등록 성문과 가장 잘 맞는 출력을 고른다."""
        # source_b에 정확히 대응하는 임베딩을 반환하는 가짜 임베더
        vectors = {id(self.source_a): [1.0, 0.0], id(self.source_b): [0.0, 1.0]}

        result = separation_svc.extract_target(
            np.zeros(100, dtype=np.float32),
            [0.0, 1.0],  # source_b 쪽을 가리키는 등록 성문
            embed_fn=lambda s: vectors[id(s)],
        )

        assert result.target_index == 1
        assert result.target_similarity == pytest.approx(1.0, abs=1e-5)
        assert np.array_equal(result.target, self.source_b)

    def test_reports_selection_margin(self):
        """1등과 2등의 차이를 보고한다 — 작으면 선택이 모호하다는 신호다."""
        vectors = {id(self.source_a): [1.0, 0.0], id(self.source_b): [0.0, 1.0]}

        result = separation_svc.extract_target(
            np.zeros(100, dtype=np.float32),
            [1.0, 0.0],
            embed_fn=lambda s: vectors[id(s)],
        )

        assert result.target_index == 0
        assert result.runner_up_similarity == pytest.approx(0.0, abs=1e-5)
        assert result.selection_margin == pytest.approx(1.0, abs=1e-5)

    def test_ambiguous_selection_has_small_margin(self):
        """두 출력이 비슷하면 마진이 작다."""
        vectors = {
            id(self.source_a): [1.0, 0.0],
            id(self.source_b): [0.99, 0.14],  # 거의 같은 방향
        }

        result = separation_svc.extract_target(
            np.zeros(100, dtype=np.float32),
            [1.0, 0.0],
            embed_fn=lambda s: vectors[id(s)],
        )

        assert result.selection_margin is not None
        assert result.selection_margin < 0.05

    def test_normalizes_reference_embedding(self):
        """정규화되지 않은 등록 성문이 들어와도 방향만 본다."""
        vectors = {id(self.source_a): [1.0, 0.0], id(self.source_b): [0.0, 1.0]}

        scaled = separation_svc.extract_target(
            np.zeros(100, dtype=np.float32),
            [0.0, 17.0],  # 크기만 다른 같은 방향
            embed_fn=lambda s: vectors[id(s)],
        )

        assert scaled.target_index == 1
        assert scaled.target_similarity == pytest.approx(1.0, abs=1e-5)

    def test_single_source_has_no_runner_up(self, monkeypatch):
        """출력이 하나뿐이면 차순위가 없다."""
        monkeypatch.setattr(
            separation_svc, "separate", lambda samples, **kw: [self.source_a]
        )

        result = separation_svc.extract_target(
            np.zeros(100, dtype=np.float32),
            [1.0, 0.0],
            embed_fn=lambda s: [1.0, 0.0],
        )

        assert result.source_count == 1
        assert result.runner_up_similarity is None
        assert result.selection_margin is None


@requires_model
class TestSepformerSeparation:
    """실제 분리 모델 동작."""

    def test_separates_two_speakers(self):
        """2인 혼합에서 두 개의 소스를 낸다."""
        a = synth_speech(3.0, f0=110.0, seed=1)
        b = synth_speech(3.0, f0=230.0, seed=2)
        mix = (a + b) / 2

        sources = separation_svc.separate(mix)

        assert len(sources) == 2
        for s in sources:
            assert s.dtype == np.float32
            assert len(s) == len(mix)
            assert np.all(np.isfinite(s))

    def test_separation_recovers_distinct_sources(self):
        """분리 출력이 서로 다르다 — 같은 신호를 두 번 뱉으면 분리가 아니다."""
        a = synth_speech(3.0, f0=110.0, seed=1)
        b = synth_speech(3.0, f0=230.0, seed=2)
        mix = (a + b) / 2

        s0, s1 = separation_svc.separate(mix)

        correlation = abs(
            float(np.dot(_unit(s0), _unit(s1)))
        )
        assert correlation < 0.9, "두 출력이 거의 같다 — 분리되지 않았다"

    def test_is_deterministic(self):
        """같은 입력은 같은 분리 결과를 낸다."""
        mix = (synth_speech(2.0, f0=110.0, seed=1) + synth_speech(2.0, f0=230.0, seed=2)) / 2

        first = separation_svc.separate(mix)
        second = separation_svc.separate(mix)

        for a, b in zip(first, second):
            assert np.allclose(a, b, atol=1e-5)


class TestConfig:
    """설정 기본값 — 분리는 기본 비활성이어야 한다."""

    def test_separation_disabled_by_default(self):
        from app.config import Settings

        settings = Settings()

        # 단일 화자 오디오에 분리를 걸면 아티팩트만 더한다 (02 §4.3)
        assert settings.separation_enabled is False

    def test_separation_model_is_16k(self):
        """16kHz 모델이어야 리샘플링 왕복이 없다."""
        from app.config import Settings

        assert "16k" in Settings().separation_model
