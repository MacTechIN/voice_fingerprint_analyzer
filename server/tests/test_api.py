"""API 통합 테스트 — 실제 파이프라인을 그대로 통과시킨다."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import ErrorCode

EXTRACT = "/api/v1/extract"


def test_health_reports_loaded_model(client):
    res = client.get("/api/v1/health")

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True  # lifespan 워밍업이 끝난 상태
    assert "ecapa" in body["model"].lower()


def test_extract_returns_embedding_and_audio_info(client, speech_wav_bytes):
    """정상 경로 — 192차원 임베딩과 전처리 요약을 반환한다."""
    res = client.post(EXTRACT, files={"file": ("speech.wav", speech_wav_bytes, "audio/wav")})

    assert res.status_code == 200
    body = res.json()

    assert body["status"] == "success"
    assert body["embedding"]["dim"] == 192
    assert len(body["embedding"]["vector"]) == 192
    assert body["embedding"]["l2_normalized"] is True
    assert np.linalg.norm(body["embedding"]["vector"]) == pytest.approx(1.0, abs=1e-5)

    audio = body["audio"]
    assert audio["sample_rate"] == 16_000
    assert audio["source_channels"] == 1
    assert audio["speech_duration_sec"] >= 1.5
    assert 0.0 < audio["speech_ratio"] <= 1.0
    assert len(audio["segments"]) >= 1
    assert body["elapsed_ms"] > 0


def test_extract_accepts_offspec_input(client, stereo_44k_wav_bytes):
    """규격을 벗어난 44.1kHz 스테레오도 서버가 흡수한다."""
    res = client.post(EXTRACT, files={"file": ("odd.wav", stereo_44k_wav_bytes, "audio/wav")})

    assert res.status_code == 200
    audio = res.json()["audio"]
    assert audio["source_sample_rate"] == 44_100
    assert audio["source_channels"] == 2
    assert audio["sample_rate"] == 16_000  # 내부 표준으로 정규화됨


def test_extract_rejects_silence_with_reason_code(client, silence_wav_bytes):
    """무음은 422 + no_speech_detected — 클라이언트가 재녹음을 안내할 근거."""
    res = client.post(EXTRACT, files={"file": ("silence.wav", silence_wav_bytes, "audio/wav")})

    assert res.status_code == 422
    body = res.json()
    assert body["status"] == "error"
    assert body["code"] == ErrorCode.NO_SPEECH_DETECTED.value
    assert body["detail"]


def test_extract_rejects_short_speech_with_reason_code(client, short_speech_wav_bytes):
    """발화가 짧으면 422 + 사유 코드."""
    res = client.post(EXTRACT, files={"file": ("short.wav", short_speech_wav_bytes, "audio/wav")})

    assert res.status_code == 422
    assert res.json()["code"] in (
        ErrorCode.SPEECH_TOO_SHORT.value,
        ErrorCode.NO_SPEECH_DETECTED.value,
    )


def test_extract_rejects_unreadable_file(client):
    """오디오가 아닌 파일은 422 + unreadable_audio."""
    res = client.post(EXTRACT, files={"file": ("x.wav", b"definitely not audio", "audio/wav")})

    assert res.status_code == 422
    assert res.json()["code"] == ErrorCode.UNREADABLE_AUDIO.value


def test_extract_requires_file(client):
    """파일 없이 호출하면 FastAPI 검증에서 422."""
    assert client.post(EXTRACT).status_code == 422


def test_same_audio_yields_same_embedding(client, speech_wav_bytes):
    """동일 오디오는 동일 임베딩 — 검증 결과의 재현성 근거."""
    files = {"file": ("speech.wav", speech_wav_bytes, "audio/wav")}
    first = client.post(EXTRACT, files=files).json()["embedding"]["vector"]
    second = client.post(EXTRACT, files=files).json()["embedding"]["vector"]

    assert np.allclose(first, second, atol=1e-6)


def test_response_schema_is_extensible(client, speech_wav_bytes):
    """Phase B~D 필드 추가에 대비해 응답이 중첩 객체 구조인지 확인한다 (01 §2).

    최상위에 값을 평평하게 늘어놓으면 필드가 늘수록 계약이 지저분해진다.
    audio/embedding으로 묶어두면 normalized_score·spoof_score를 최상위에
    무중단으로 덧붙일 수 있다.
    """
    body = client.post(
        EXTRACT, files={"file": ("speech.wav", speech_wav_bytes, "audio/wav")}
    ).json()

    assert isinstance(body["audio"], dict)
    assert isinstance(body["embedding"], dict)
    assert set(body) >= {"status", "audio", "embedding", "elapsed_ms"}
