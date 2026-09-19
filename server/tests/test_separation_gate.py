"""분리 게이트 동작 테스트 (Phase 7 후속).

게이트는 "원본이 이미 등록 화자와 충분히 닮았으면 분리를 건너뛴다"는 규칙이다.
분리 모델 없이도 규칙 자체는 검증할 수 있으므로, 분리를 가짜로 대체하고
**호출되었는지 여부**를 본다.

건너뛰는 경로에 추가 비용이 없어야 한다는 점도 함께 본다. 게이트 판정에 쓴
임베딩을 버리고 다시 뽑으면 게이트가 아끼려던 비용의 일부를 도로 쓰게 된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from app.api import routes
from app.config import Settings
from app.services import separation as separation_svc

from .conftest import SAMPLE_RATE, _write_wav, synth_speech


@dataclass
class _FakeResult:
    target: np.ndarray
    source_count: int = 2
    target_index: int = 0
    target_similarity: float = 0.9
    selection_margin: float | None = 0.3


@pytest.fixture
def audio() -> bytes:
    return _write_wav(synth_speech(4.0, f0=120.0, seed=3), SAMPLE_RATE)


@pytest.fixture
def spy(monkeypatch):
    """분리를 가짜로 대체하고 호출 횟수를 센다."""
    calls: list[dict] = []

    def fake_extract_target(samples, reference, **kw):
        calls.append({"len": len(samples)})
        # 분리 결과를 원본과 다르게 만들어, 최종 임베딩이 어느 경로에서
        # 나왔는지 구분할 수 있게 한다.
        return _FakeResult(target=synth_speech(4.0, f0=260.0, seed=9))

    monkeypatch.setattr(separation_svc, "extract_target", fake_extract_target)
    return calls


def _settings(**kw) -> Settings:
    return Settings(separation_enabled=True, **kw)


def _reference_from(audio: bytes, settings: Settings) -> list[float]:
    """같은 오디오의 임베딩 — 게이트 점수가 1에 가깝게 나오는 기준."""
    return routes._analyze(audio, settings).embedding.vector


class TestGateDecision:
    def test_no_gate_always_separates(self, audio, spy):
        """게이트가 없으면(None) 기존 동작 그대로 항상 분리한다."""
        settings = _settings(separation_gate_score=None)
        ref = _reference_from(audio, Settings())

        result = routes._analyze(audio, settings, reference_embeddings=[ref])

        assert len(spy) == 1
        assert result.separation is not None
        assert result.separation.applied is True
        assert result.separation.gate_score is None

    def test_similar_audio_skips_separation(self, audio, spy):
        """원본이 등록 성문과 닮았으면 분리를 건너뛴다."""
        settings = _settings(separation_gate_score=0.5)
        ref = _reference_from(audio, Settings())

        result = routes._analyze(audio, settings, reference_embeddings=[ref])

        assert spy == [], "닮은 오디오인데 분리를 호출했다"
        assert result.separation is not None
        assert result.separation.applied is False
        assert result.separation.gate_score == pytest.approx(1.0, abs=0.05)
        assert result.separation.gate_threshold == 0.5

    def test_unrelated_audio_runs_separation(self, audio, spy):
        """등록 성문과 닮지 않았으면 분리한다 — 혼합일 수 있기 때문이다."""
        settings = _settings(separation_gate_score=0.5)
        # 임의의 방향 벡터. 실제 화자 임베딩과 상관이 없다.
        rng = np.random.default_rng(0)
        dim = len(_reference_from(audio, Settings()))
        ref = list(rng.normal(size=dim).astype(float))

        result = routes._analyze(audio, settings, reference_embeddings=[ref])

        assert len(spy) == 1
        assert result.separation is not None
        assert result.separation.applied is True
        # 게이트를 거쳤으므로 판정 근거가 남아야 한다.
        assert result.separation.gate_score is not None
        assert result.separation.gate_score < 0.5

    def test_threshold_boundary_is_inclusive(self, audio, spy):
        """임계값과 같으면 건너뛴다 — 경계에서 분리가 켜지지 않아야 한다."""
        settings_probe = Settings()
        ref = _reference_from(audio, settings_probe)
        baseline = routes._analyze(
            audio, _settings(separation_gate_score=1.1), reference_embeddings=[ref]
        )
        measured = baseline.separation.gate_score
        spy.clear()

        result = routes._analyze(
            audio, _settings(separation_gate_score=measured), reference_embeddings=[ref]
        )

        assert spy == []
        assert result.separation.applied is False


class TestGateCost:
    def test_skipped_path_reuses_gate_embedding(self, audio, spy, monkeypatch):
        """건너뛸 때 임베딩을 다시 뽑지 않는다.

        게이트 판정용으로 뽑은 임베딩을 버리고 최종 판정에서 또 뽑으면,
        게이트가 아끼려던 비용을 도로 쓰게 된다.
        """
        from app.services import embedding as embedding_svc

        original = embedding_svc.extract
        count = {"n": 0}

        def counting_extract(*args, **kw):
            count["n"] += 1
            return original(*args, **kw)

        settings = _settings(separation_gate_score=0.5)
        ref = _reference_from(audio, Settings())

        monkeypatch.setattr(embedding_svc, "extract", counting_extract)
        routes._analyze(audio, settings, reference_embeddings=[ref])

        assert spy == []
        assert count["n"] == 1, f"임베딩을 {count['n']}번 뽑았다 — 재사용되지 않았다"


class TestMultipleEnrollments:
    def test_gate_uses_best_matching_enrollment(self, audio, spy):
        """등록 성문이 여럿이면 가장 잘 맞는 것을 기준으로 판단한다.

        최종 판정이 최대 유사도 기준이므로 게이트도 같아야 한다. 최근 것 하나만
        보면, 다른 성문과 잘 맞는 요청까지 분리를 돌려 비용을 버린다.
        """
        settings = _settings(separation_gate_score=0.5)
        ref_match = _reference_from(audio, Settings())
        rng = np.random.default_rng(1)
        ref_unrelated = list(rng.normal(size=len(ref_match)).astype(float))

        # 최근 것(첫 번째)은 맞지 않고 두 번째가 맞는 경우
        result = routes._analyze(
            audio, settings, reference_embeddings=[ref_unrelated, ref_match]
        )

        assert spy == [], "잘 맞는 성문이 있는데 분리를 호출했다"
        assert result.separation.applied is False
        assert result.separation.gate_score == pytest.approx(1.0, abs=0.05)

    def test_target_selection_uses_most_recent(self, audio, spy):
        """분리할 때 타겟 선택 기준은 가장 최근 성문이다."""
        settings = _settings(separation_gate_score=None)
        ref_a = _reference_from(audio, Settings())
        ref_b = [0.0] * len(ref_a)

        routes._analyze(audio, settings, reference_embeddings=[ref_a, ref_b])

        assert len(spy) == 1


class TestGateGuards:
    def test_no_reference_means_no_gate(self, audio, spy):
        """등록 성문이 없으면 게이트도 분리도 동작하지 않는다.

        비교 대상이 없으면 게이트 점수 자체를 낼 수 없다.
        """
        settings = _settings(separation_gate_score=0.5)

        result = routes._analyze(audio, settings, reference_embeddings=None)

        assert spy == []
        assert result.separation is None

    def test_separation_disabled_ignores_gate(self, audio, spy):
        """분리가 꺼져 있으면 게이트 설정이 있어도 아무것도 하지 않는다."""
        settings = Settings(separation_enabled=False, separation_gate_score=0.5)
        ref = _reference_from(audio, Settings())

        result = routes._analyze(audio, settings, reference_embeddings=[ref])

        assert spy == []
        assert result.separation is None


class TestGateSetting:
    """환경변수 파싱 — 잘못 두면 서버가 부팅에 실패한다."""

    def test_blank_env_disables_gate(self, monkeypatch):
        """빈 값은 "게이트 없음"이다.

        배포 템플릿에서 `VG_SEPARATION_GATE_SCORE=`로 비워 두는 것은 흔한
        표기인데, 그대로 두면 float 파싱 실패로 기동이 막힌다.
        """
        for raw in ("", "   "):
            monkeypatch.setenv("VG_SEPARATION_GATE_SCORE", raw)
            assert Settings().separation_gate_score is None

    def test_numeric_env_is_parsed(self, monkeypatch):
        monkeypatch.setenv("VG_SEPARATION_GATE_SCORE", "0.45")
        assert Settings().separation_gate_score == pytest.approx(0.45)

    def test_garbage_env_fails_loudly(self, monkeypatch):
        """숫자가 아닌 값은 조용히 무시하지 않는다.

        오타를 None으로 삼키면 게이트가 꺼진 줄 모르고 운영하게 된다.
        """
        monkeypatch.setenv("VG_SEPARATION_GATE_SCORE", "high")
        with pytest.raises(Exception):
            Settings()

