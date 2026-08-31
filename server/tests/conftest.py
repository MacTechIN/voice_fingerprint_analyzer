"""테스트 공용 픽스처.

실제 모델(Silero VAD, ECAPA-TDNN)을 그대로 사용한다. 임베딩 품질과 VAD 동작이
이 파이프라인의 본질이므로, 모킹하면 정작 검증하고 싶은 것이 빠진다.
모델 적재는 세션 스코프로 한 번만 한다.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

SAMPLE_RATE = 16_000


def _write_wav(samples: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def synth_speech(duration_sec: float, *, f0: float = 120.0, seed: int = 0) -> np.ndarray:
    """유성음을 흉내 낸 합성 파형.

    실제 음성이 아니므로 화자 동일성 판정에는 쓸 수 없지만, VAD가 "발화 있음"으로
    판정하기에 충분한 하모닉 구조와 진폭 변조를 갖는다. VAD·임베딩 파이프라인의
    배관(plumbing)을 검증하는 것이 목적이다.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)

    # 하모닉 스택 — 사람 목소리의 기본 구조
    wave = np.zeros_like(t)
    for harmonic in range(1, 12):
        wave += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)

    # 음절 리듬을 흉내 낸 진폭 변조 (4Hz)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t))
    wave *= envelope

    # 자음 성분에 해당하는 약한 잡음
    wave += 0.02 * rng.standard_normal(len(t))

    peak = np.abs(wave).max()
    if peak > 0:
        wave = wave / peak * 0.8
    return wave.astype(np.float32)


@pytest.fixture(scope="session")
def speech_wav_bytes() -> bytes:
    """VAD를 통과할 만큼 충분히 긴 발화 (3초)."""
    return _write_wav(synth_speech(3.0))


@pytest.fixture(scope="session")
def short_speech_wav_bytes() -> bytes:
    """유효 발화가 하한(1.5초)에 못 미치는 짧은 오디오."""
    return _write_wav(synth_speech(0.4))


@pytest.fixture(scope="session")
def silence_wav_bytes() -> bytes:
    """발화가 전혀 없는 무음 (미세한 배경 잡음만)."""
    rng = np.random.default_rng(1)
    samples = (0.0005 * rng.standard_normal(SAMPLE_RATE * 3)).astype(np.float32)
    return _write_wav(samples)


@pytest.fixture(scope="session")
def stereo_44k_wav_bytes() -> bytes:
    """규격을 벗어난 입력 — 44.1kHz 스테레오. 서버가 흡수해야 한다."""
    rate = 44_100
    t = np.linspace(0, 3.0, int(rate * 3.0), endpoint=False)
    wave = np.zeros_like(t)
    for harmonic in range(1, 12):
        wave += (1.0 / harmonic) * np.sin(2 * np.pi * 120.0 * harmonic * t)
    wave *= 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t))
    peak = np.abs(wave).max()
    wave = (wave / peak * 0.8).astype(np.float32)
    stereo = np.stack([wave, wave], axis=1)
    return _write_wav(stereo, rate)


@pytest.fixture(scope="session")
def client():
    """모델을 적재한 테스트 클라이언트 (세션 1회).

    저장소는 인메모리로 뜬다 (`VG_DATABASE_URL` 미설정). Postgres 경로는
    `test_repository.py`가 실제 DB로 따로 검증한다.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def fresh_storage(client):
    """각 테스트가 빈 저장소에서 시작하도록 초기화한다.

    세션 스코프 클라이언트를 공유하므로, 앞선 테스트가 등록한 성문이 남아 있으면
    "미등록 사용자" 같은 조건을 만들 수 없다.
    """
    from app.db import session as db_session
    from app.db.repository import InMemoryRepository

    repo = db_session.get_repository()
    assert isinstance(repo, InMemoryRepository), "테스트는 인메모리 저장소를 전제한다"
    repo._rows.clear()
    repo._logs.clear()
    repo._next_id = 1
    return repo


def unique_user(prefix: str = "user") -> str:
    """테스트 간 충돌하지 않는 사용자 ID."""
    import uuid

    return f"{prefix}-{uuid.uuid4().hex[:8]}"
