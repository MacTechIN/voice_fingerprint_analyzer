"""VAD 단위 테스트 — 실제 Silero 모델 사용."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import AudioRejected, ErrorCode
from app.services import vad as vad_svc

from .conftest import SAMPLE_RATE, synth_speech

VAD_KWARGS = dict(threshold=0.5, min_silence_ms=300, speech_pad_ms=30, min_speech_sec=1.5)


def test_vad_strips_silence_around_speech():
    """앞뒤 무음을 제거하고 발화 구간만 남긴다."""
    silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    speech = synth_speech(3.0)
    padded = np.concatenate([silence, speech, silence])

    result = vad_svc.apply(padded, SAMPLE_RATE, **VAD_KWARGS)

    assert result.total_duration_sec == pytest.approx(7.0, abs=0.05)
    # 무음 4초가 제거되어 발화 길이는 전체보다 뚜렷하게 짧아야 한다
    assert result.speech_duration_sec < result.total_duration_sec - 3.0
    assert result.speech_ratio < 0.7
    assert len(result.segments) >= 1
    # 검출 구간은 무음 2초 이후에서 시작해야 한다
    assert result.segments[0].start >= 1.5


def test_vad_rejects_pure_silence():
    """발화가 없으면 반려한다."""
    rng = np.random.default_rng(1)
    samples = (0.0005 * rng.standard_normal(SAMPLE_RATE * 3)).astype(np.float32)

    with pytest.raises(AudioRejected) as exc:
        vad_svc.apply(samples, SAMPLE_RATE, **VAD_KWARGS)
    assert exc.value.code is ErrorCode.NO_SPEECH_DETECTED


def test_vad_rejects_speech_below_minimum():
    """유효 발화가 하한 미달이면 반려한다."""
    short = synth_speech(0.4)

    with pytest.raises(AudioRejected) as exc:
        vad_svc.apply(short, SAMPLE_RATE, **VAD_KWARGS)
    # 너무 짧으면 아예 검출이 안 될 수도, 검출되어도 길이 미달일 수도 있다.
    # 둘 다 클라이언트에게는 "다시 녹음하세요"로 이어지는 정당한 반려다.
    assert exc.value.code in (ErrorCode.SPEECH_TOO_SHORT, ErrorCode.NO_SPEECH_DETECTED)


def test_vad_segments_are_ordered_and_within_bounds():
    """검출 구간은 시간순이며 오디오 범위를 벗어나지 않는다."""
    speech = synth_speech(4.0)
    result = vad_svc.apply(speech, SAMPLE_RATE, **VAD_KWARGS)

    prev_end = 0.0
    for seg in result.segments:
        assert 0.0 <= seg.start < seg.end <= result.total_duration_sec
        assert seg.start >= prev_end  # 구간이 겹치거나 역행하지 않는다
        prev_end = seg.end
