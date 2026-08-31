"""딥페이크 탐지 테스트 (Phase 8).

가중치 파일이 없으면 모델을 쓰는 테스트는 건너뛴다 — 저장소에 바이너리를
커밋하지 않기 때문이다.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from app.services import antispoof

from .conftest import SAMPLE_RATE, synth_speech

WEIGHTS = "./.model_cache/AASIST-L.pth"
HEAVY = os.environ.get("VG_TEST_HEAVY") == "1"

requires_weights = pytest.mark.skipif(
    not (HEAVY or Path(WEIGHTS).exists()),
    reason="AASIST-L 가중치 없음 (eval/README.md 준비 절차 참조)",
)


class TestSegmentation:
    """구간 분할 — 모델 없이 검증할 수 있는 부분."""

    def test_short_audio_is_padded_by_repetition(self):
        """모델 입력 길이보다 짧으면 반복해 채운다 (원 저자 구현과 동일)."""
        short = np.arange(1000, dtype=np.float32)

        chunks = antispoof._segments(short)

        assert len(chunks) == 1
        assert len(chunks[0]) == antispoof.NB_SAMP
        # 반복이므로 앞부분이 원본과 같아야 한다
        assert np.array_equal(chunks[0][:1000], short)

    def test_exact_length_gives_one_chunk(self):
        exact = np.zeros(antispoof.NB_SAMP, dtype=np.float32)

        assert len(antispoof._segments(exact)) == 1

    def test_long_audio_is_split(self):
        """긴 오디오는 여러 구간으로 나눈다 — 일부만 합성인 공격을 잡기 위해서다."""
        long_audio = np.zeros(antispoof.NB_SAMP * 3, dtype=np.float32)

        chunks = antispoof._segments(long_audio)

        assert len(chunks) == 3
        assert all(len(c) == antispoof.NB_SAMP for c in chunks)

    def test_large_remainder_adds_tail_chunk(self):
        """꼬리가 절반 이상 남으면 끝에서 한 구간을 더 본다."""
        audio = np.zeros(int(antispoof.NB_SAMP * 1.8), dtype=np.float32)

        chunks = antispoof._segments(audio)

        assert len(chunks) == 2, "남는 0.8구간을 그냥 버리면 그 부분을 검사하지 못한다"

    def test_small_remainder_is_dropped(self):
        """꼬리가 짧으면 추가 구간을 만들지 않는다 — 거의 중복이라 의미가 없다."""
        audio = np.zeros(int(antispoof.NB_SAMP * 1.2), dtype=np.float32)

        assert len(antispoof._segments(audio)) == 1


@requires_weights
class TestDetection:
    """실제 모델 동작."""

    def test_detects_obvious_synthetic_tone(self):
        """명백한 합성 신호를 위조로 판정한다."""
        tone = synth_speech(4.0, f0=140.0)

        result = antispoof.detect(tone, weights_path=WEIGHTS, threshold=0.5)

        assert result.is_spoof is True
        assert result.spoof_score > 0.5
        assert result.bonafide_score == pytest.approx(1 - result.spoof_score)

    def test_threshold_controls_verdict(self):
        """임계값이 판정을 가른다."""
        tone = synth_speech(4.0, f0=140.0)

        strict = antispoof.detect(tone, weights_path=WEIGHTS, threshold=0.1)
        lenient = antispoof.detect(tone, weights_path=WEIGHTS, threshold=0.999999)

        assert strict.spoof_score == pytest.approx(lenient.spoof_score)
        assert strict.is_spoof is True
        # 같은 점수라도 임계값에 따라 판정이 달라진다
        assert lenient.is_spoof == (lenient.spoof_score >= 0.999999)

    def test_score_is_bounded_probability(self):
        tone = synth_speech(4.0)

        result = antispoof.detect(tone, weights_path=WEIGHTS, threshold=0.5)

        assert 0.0 <= result.spoof_score <= 1.0

    def test_is_deterministic(self):
        """같은 입력은 같은 점수를 낸다 — 판정이 재현되지 않으면 감사가 무의미하다."""
        tone = synth_speech(4.0, f0=120.0)

        first = antispoof.detect(tone, weights_path=WEIGHTS, threshold=0.5)
        second = antispoof.detect(tone, weights_path=WEIGHTS, threshold=0.5)

        assert first.spoof_score == pytest.approx(second.spoof_score, abs=1e-6)

    def test_reports_segment_count(self):
        long_audio = synth_speech(12.0)

        result = antispoof.detect(long_audio, weights_path=WEIGHTS, threshold=0.5)

        assert result.segments_scored >= 2

    def test_max_aggregation_catches_partial_spoof(self):
        """일부 구간만 합성이어도 잡는다.

        평균을 쓰면 진짜 음성 구간이 합성 구간을 희석해 놓친다. 최댓값을 쓰는
        이유가 이것이다.
        """
        real_ish = np.random.default_rng(0).normal(0, 0.05, antispoof.NB_SAMP).astype(np.float32)
        synthetic = synth_speech(4.1, f0=140.0)[: antispoof.NB_SAMP]
        mixed = np.concatenate([real_ish, real_ish, synthetic])

        result = antispoof.detect(mixed, weights_path=WEIGHTS, threshold=0.5)

        assert result.segments_scored >= 3
        assert result.spoof_score > 0.5, "합성 구간이 섞였는데 놓쳤다"


class TestConfig:
    def test_disabled_by_default(self):
        """가중치를 내려받지 않은 환경에서 기동이 실패하지 않도록 기본 비활성이다."""
        from app.config import Settings

        assert Settings().antispoof_enabled is False

    def test_error_code_is_stable(self):
        """반려 코드는 클라이언트 계약이다 — 값이 바뀌면 앱이 깨진다."""
        from app.core.errors import ErrorCode

        assert ErrorCode.SPOOF_DETECTED.value == "spoof_detected"
