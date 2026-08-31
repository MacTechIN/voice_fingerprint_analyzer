"""오디오 디코딩 단위 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import AudioRejected, ErrorCode
from app.services import audio as audio_svc


def test_decode_normalizes_to_target_rate_and_mono(stereo_44k_wav_bytes):
    """44.1kHz 스테레오 입력을 16kHz 모노로 정규화한다."""
    decoded = audio_svc.decode(stereo_44k_wav_bytes, target_rate=16_000, max_sec=300)

    assert decoded.sample_rate == 16_000
    assert decoded.source_sample_rate == 44_100
    assert decoded.source_channels == 2
    assert decoded.samples.ndim == 1
    assert decoded.samples.dtype == np.float32
    assert decoded.duration_sec == pytest.approx(3.0, abs=0.05)


def test_decode_preserves_16k_mono_untouched(speech_wav_bytes):
    """규격에 맞는 입력은 리샘플링 없이 통과한다."""
    decoded = audio_svc.decode(speech_wav_bytes, target_rate=16_000, max_sec=300)

    assert decoded.sample_rate == decoded.source_sample_rate == 16_000
    assert decoded.source_channels == 1
    assert len(decoded.samples) == 16_000 * 3


def test_decode_rejects_empty_body():
    with pytest.raises(AudioRejected) as exc:
        audio_svc.decode(b"", target_rate=16_000, max_sec=300)
    assert exc.value.code is ErrorCode.EMPTY_FILE


def test_decode_rejects_garbage():
    """오디오가 아닌 바이트는 디코딩 불가로 반려한다."""
    with pytest.raises(AudioRejected) as exc:
        audio_svc.decode(b"not an audio file at all", target_rate=16_000, max_sec=300)
    assert exc.value.code is ErrorCode.UNREADABLE_AUDIO


def test_decode_rejects_overlong_audio(speech_wav_bytes):
    """허용 길이를 넘으면 반려한다 (긴 오디오는 Phase 7 청크 처리 대상)."""
    with pytest.raises(AudioRejected) as exc:
        audio_svc.decode(speech_wav_bytes, target_rate=16_000, max_sec=1.0)
    assert exc.value.code is ErrorCode.AUDIO_TOO_LONG
