"""Silero VAD 기반 발화 구간 검출.

무음·소음 구간을 제거하지 않으면 노이즈가 성문 벡터를 오염시키므로 파이프라인
최전단에 둔다 (02 §2.2). 모델은 프로세스당 1회 적재해 재사용한다.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from app.core.errors import AudioRejected, ErrorCode

logger = logging.getLogger(__name__)


@contextmanager
def _preserve_torch_threads():
    """블록 안에서 바뀐 torch 스레드 수를 되돌린다.

    silero-vad가 프로세스 전역 설정을 건드리는 것을 격리하기 위한 장치다.
    라이브러리가 전역 상태를 바꾸는 것은 우리가 고칠 수 없으므로, 경계에서
    막는다.
    """
    before = torch.get_num_threads()
    try:
        yield
    finally:
        if torch.get_num_threads() != before:
            torch.set_num_threads(before)

_model = None
_get_speech_timestamps = None
_load_lock = threading.Lock()


def _ensure_loaded() -> None:
    """VAD 모델을 지연 적재한다 (프로세스당 1회)."""
    global _model, _get_speech_timestamps
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:  # 락 대기 중 다른 스레드가 적재했을 수 있다
            return
        from silero_vad import get_speech_timestamps, load_silero_vad

        logger.info("Silero VAD 모델 적재 중")
        model = load_silero_vad()
        _get_speech_timestamps = get_speech_timestamps
        _model = model
        logger.info("Silero VAD 모델 적재 완료")


@dataclass(frozen=True)
class SpeechSegment:
    """발화 구간 (초 단위)."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class VadResult:
    """VAD 적용 결과."""

    samples: np.ndarray
    """발화 구간만 이어 붙인 파형."""

    segments: tuple[SpeechSegment, ...]
    speech_duration_sec: float
    total_duration_sec: float

    @property
    def speech_ratio(self) -> float:
        if self.total_duration_sec <= 0:
            return 0.0
        return self.speech_duration_sec / self.total_duration_sec


def apply(
    samples: np.ndarray,
    sample_rate: int,
    *,
    threshold: float,
    min_silence_ms: int,
    speech_pad_ms: int,
    min_speech_sec: float,
) -> VadResult:
    """발화 구간을 검출해 이어 붙인 파형을 반환한다.

    Raises:
        AudioRejected: 발화가 없거나 유효 발화 길이가 하한에 미달한 경우.
    """
    total_duration = len(samples) / sample_rate

    # silero-vad는 적재·실행 과정에서 torch.set_num_threads(1)을 호출해
    # **프로세스 전체**를 단일 스레드로 낮춘다. VAD는 가벼워 스스로는 문제가
    # 없지만, 같은 프로세스의 음성 분리 추론이 4배 느려진다(6초 오디오 기준
    # 7.3초 → 28초). VAD 전후로 스레드 수를 보존한다.
    with _preserve_torch_threads():
        _ensure_loaded()
        timestamps = _get_speech_timestamps(
            torch.from_numpy(samples),
            _model,
            sampling_rate=sample_rate,
            threshold=threshold,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=False,
        )

    if not timestamps:
        raise AudioRejected(
            ErrorCode.NO_SPEECH_DETECTED,
            "음성이 감지되지 않았습니다. 조용한 곳에서 다시 녹음해주세요.",
            total_duration_sec=total_duration,
        )

    chunks = [samples[ts["start"] : ts["end"]] for ts in timestamps]
    speech = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    speech_duration = len(speech) / sample_rate

    if speech_duration < min_speech_sec:
        raise AudioRejected(
            ErrorCode.SPEECH_TOO_SHORT,
            (
                f"유효 발화가 너무 짧습니다 ({speech_duration:.1f}초). "
                f"{min_speech_sec:.1f}초 이상 말해주세요."
            ),
            speech_duration_sec=speech_duration,
            total_duration_sec=total_duration,
        )

    segments = tuple(
        SpeechSegment(start=ts["start"] / sample_rate, end=ts["end"] / sample_rate)
        for ts in timestamps
    )
    return VadResult(
        samples=speech,
        segments=segments,
        speech_duration_sec=speech_duration,
        total_duration_sec=total_duration,
    )
