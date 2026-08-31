"""오디오 디코딩 및 정규화.

서버는 16kHz/Mono/float32를 내부 표준으로 삼는다 (02 §1). 클라이언트가 규격을
지키는 것이 원칙이지만, 규격을 벗어난 입력도 여기서 흡수해 파이프라인 뒤쪽이
샘플레이트·채널 수를 신경 쓰지 않도록 한다.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from app.core.errors import AudioRejected, ErrorCode


@dataclass(frozen=True)
class DecodedAudio:
    """디코딩된 모노 파형."""

    samples: np.ndarray
    """float32, shape (N,), 범위 [-1, 1]."""

    sample_rate: int
    source_sample_rate: int
    source_channels: int

    @property
    def duration_sec(self) -> float:
        return len(self.samples) / self.sample_rate


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """선형 보간 리샘플링.

    torchaudio의 sinc 보간보다 품질이 낮지만, 정규 경로(클라이언트가 16kHz로
    보내는 경우)에서는 호출되지 않는 예외 처리용이므로 의존성을 늘리지 않는다.
    규격을 벗어난 입력이 임베딩 품질에 영향을 준다면 클라이언트를 고치는 것이 맞다.
    """
    if src_rate == dst_rate:
        return samples
    duration = len(samples) / src_rate
    dst_len = int(round(duration * dst_rate))
    if dst_len <= 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.linspace(0, len(samples) - 1, num=dst_len, dtype=np.float64)
    return np.interp(src_idx, np.arange(len(samples)), samples).astype(np.float32)


def decode(raw: bytes, target_rate: int, max_sec: float) -> DecodedAudio:
    """업로드 바이트를 모노 float32 파형으로 디코딩한다.

    Raises:
        AudioRejected: 디코딩 불가하거나 허용 길이를 초과한 경우.
    """
    if not raw:
        raise AudioRejected(ErrorCode.EMPTY_FILE, "업로드된 오디오가 비어 있습니다.")

    try:
        samples, src_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile은 포맷별로 다른 예외를 던진다
        raise AudioRejected(
            ErrorCode.UNREADABLE_AUDIO,
            "오디오를 디코딩할 수 없습니다. 16kHz 16-bit Mono WAV로 보내주세요.",
        ) from exc

    src_channels = samples.shape[1]
    mono = samples.mean(axis=1) if src_channels > 1 else samples[:, 0]

    duration = len(mono) / src_rate if src_rate else 0.0
    if duration > max_sec:
        raise AudioRejected(
            ErrorCode.AUDIO_TOO_LONG,
            f"오디오가 너무 깁니다 ({duration:.1f}초). 최대 {max_sec:.0f}초까지 처리합니다.",
            duration_sec=duration,
        )

    resampled = _resample(np.ascontiguousarray(mono, dtype=np.float32), src_rate, target_rate)
    return DecodedAudio(
        samples=resampled,
        sample_rate=target_rate,
        source_sample_rate=src_rate,
        source_channels=src_channels,
    )
