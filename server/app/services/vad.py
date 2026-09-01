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

# silero-vad의 TorchScript 모델은 **내부에 RNN 상태를 들고 있다**(`self._state`).
# 하나의 인스턴스를 여러 스레드가 동시에 쓰면 상태가 깨져 추론이 실패한다 —
# 실측: 동시 요청 4건 중 2건이 500으로 떨어졌다.
#
# 스레드마다 별도 인스턴스를 둬 격리한다. 락으로 직렬화하면 VAD가 파이프라인의
# 60%를 차지하는 만큼 처리량이 그대로 막히고, 모델이 약 2MB라 스레드당 비용은
# 작다.
_thread_local = threading.local()
_get_speech_timestamps = None
_load_lock = threading.Lock()


#: ONNX 런타임을 쓸지 여부. JIT 대비 2배 빠르고(5초 오디오 53.2ms → 26.0ms)
#: 검출 결과는 동일하다(LibriSpeech 6개 발화에서 구간 수·길이 완전 일치).
#: VAD가 파이프라인의 약 60%를 차지하므로 체감 차이가 크다.
USE_ONNX = True


def _ensure_loaded():
    """호출 스레드용 VAD 모델을 지연 적재한다.

    모델 파일은 최초 1회만 내려받고 이후 캐시를 읽으므로, 스레드별 적재는
    디스크에서 모델을 읽는 정도로 빠르다.
    """
    global _get_speech_timestamps

    model = getattr(_thread_local, "model", None)
    if model is not None:
        return model

    # 적재 자체는 라이브러리 전역 상태를 건드리므로 직렬화한다
    with _load_lock:
        from silero_vad import get_speech_timestamps, load_silero_vad

        logger.info(
            "Silero VAD 모델 적재 중 (스레드 %s, %s)",
            threading.current_thread().name,
            "ONNX" if USE_ONNX else "JIT",
        )
        model = load_silero_vad(onnx=USE_ONNX)
        _get_speech_timestamps = get_speech_timestamps

    _thread_local.model = model
    return model


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
        model = _ensure_loaded()
        # 직전 요청이 남긴 RNN 상태가 이번 판정에 섞이지 않게 초기화한다.
        # 스레드별 인스턴스를 써도 같은 스레드가 요청을 이어 받으므로 필요하다.
        if hasattr(model, "reset_states"):
            model.reset_states()
        timestamps = _get_speech_timestamps(
            torch.from_numpy(samples),
            model,
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
