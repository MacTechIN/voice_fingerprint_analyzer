"""등록·검증 통합 테스트 — 실제 파이프라인을 그대로 통과시킨다."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from app.config import get_settings
from app.core.errors import ErrorCode

from .conftest import SAMPLE_RATE, synth_speech, unique_user

ENROLL = "/api/v1/enroll"
VERIFY = "/api/v1/verify"


def _wav(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _files(data: bytes, name: str = "a.wav") -> dict:
    return {"file": (name, data, "audio/wav")}


def test_enroll_stores_voiceprint(client, fresh_storage, speech_wav_bytes):
    """등록하면 성문이 저장되고 ID가 돌아온다 (FR-04)."""
    user = unique_user()

    res = client.post(ENROLL, data={"user_id": user}, files=_files(speech_wav_bytes))

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["user_id"] == user
    assert body["enrollment_id"] > 0
    assert body["replaced"] == 0  # 첫 등록이므로 대체된 것이 없다
    assert body["embedding"]["dim"] == 192
    assert body["audio"]["speech_duration_sec"] >= 1.5

    stored = fresh_storage._rows
    assert len(stored) == 1
    assert stored[0]["user_id"] == user
    assert stored[0]["is_active"] is True


def test_reenroll_deactivates_previous(client, fresh_storage, speech_wav_bytes):
    """재등록은 기존 성문을 비활성화하되 삭제하지는 않는다."""
    user = unique_user()
    client.post(ENROLL, data={"user_id": user}, files=_files(speech_wav_bytes))

    res = client.post(ENROLL, data={"user_id": user}, files=_files(speech_wav_bytes))

    assert res.status_code == 200
    assert res.json()["replaced"] == 1

    rows = fresh_storage._rows
    assert len(rows) == 2  # 이력이 남아 있다
    assert [r["is_active"] for r in rows] == [False, True]


def test_enroll_rejects_short_speech(client, fresh_storage, short_speech_wav_bytes):
    """발화가 짧으면 등록되지 않는다 — 부실한 성문이 DB에 들어가면 안 된다."""
    user = unique_user()

    res = client.post(ENROLL, data={"user_id": user}, files=_files(short_speech_wav_bytes))

    assert res.status_code == 422
    assert res.json()["code"] in (
        ErrorCode.SPEECH_TOO_SHORT.value,
        ErrorCode.NO_SPEECH_DETECTED.value,
    )
    assert fresh_storage._rows == []


def test_enroll_requires_user_id(client, fresh_storage, speech_wav_bytes):
    assert client.post(ENROLL, files=_files(speech_wav_bytes)).status_code == 422


def test_verify_same_audio_matches(client, fresh_storage, speech_wav_bytes):
    """등록에 쓴 음성으로 검증하면 동일인으로 판정된다 (FR-05, FR-06)."""
    user = unique_user()
    client.post(ENROLL, data={"user_id": user}, files=_files(speech_wav_bytes))

    res = client.post(VERIFY, data={"user_id": user}, files=_files(speech_wav_bytes))

    assert res.status_code == 200
    body = res.json()
    assert body["is_verified"] is True
    assert body["raw_cosine"] == pytest.approx(1.0, abs=1e-4)
    assert body["match_probability"] == pytest.approx(100.0, abs=0.5)
    assert body["compared_enrollments"] == 1
    # 임계값은 캘리브레이션으로 갱신되므로 값을 고정하지 않고 설정을 참조한다
    assert body["threshold"] == pytest.approx(get_settings().match_threshold)


def test_verify_different_source_scores_lower(client, fresh_storage):
    """음향적으로 다른 소스는 유사도가 낮게 나온다.

    합성음이므로 화자 인식 정확도를 재는 것이 아니라, 유사도가 입력에 따라
    실제로 달라진다는 것 — 즉 판정이 상수가 아니라는 것 — 을 확인한다.
    """
    user = unique_user()
    enrolled = _wav(synth_speech(3.0, f0=100.0, seed=1))
    other = _wav(synth_speech(3.0, f0=230.0, seed=2))

    client.post(ENROLL, data={"user_id": user}, files=_files(enrolled))
    same = client.post(VERIFY, data={"user_id": user}, files=_files(enrolled)).json()
    diff = client.post(VERIFY, data={"user_id": user}, files=_files(other)).json()

    assert diff["raw_cosine"] < same["raw_cosine"]


def test_verify_unenrolled_user_is_rejected(client, fresh_storage, speech_wav_bytes):
    """등록 이력이 없는 사용자는 not_enrolled로 반려된다."""
    res = client.post(
        VERIFY, data={"user_id": unique_user("ghost")}, files=_files(speech_wav_bytes)
    )

    assert res.status_code == 422
    assert res.json()["code"] == ErrorCode.NOT_ENROLLED.value


def test_verify_rejects_model_mismatch(client, fresh_storage, speech_wav_bytes):
    """등록 성문이 다른 모델 것이면 비교하지 않고 재등록을 요구한다.

    조용히 비교하면 무의미한 유사도가 그대로 인증 판정에 쓰인다 (02 §6).
    """
    user = unique_user()
    client.post(ENROLL, data={"user_id": user}, files=_files(speech_wav_bytes))
    # 모델을 교체한 상황을 흉내 낸다
    fresh_storage._rows[0]["model"] = "some/other-model-v2"

    res = client.post(VERIFY, data={"user_id": user}, files=_files(speech_wav_bytes))

    assert res.status_code == 422
    body = res.json()
    assert body["code"] == ErrorCode.MODEL_MISMATCH.value
    assert "재등록" in body["detail"]


def test_verify_rejects_bad_audio(client, fresh_storage, speech_wav_bytes, silence_wav_bytes):
    """등록되어 있어도 검증 오디오가 무음이면 반려된다."""
    user = unique_user()
    client.post(ENROLL, data={"user_id": user}, files=_files(speech_wav_bytes))

    res = client.post(VERIFY, data={"user_id": user}, files=_files(silence_wav_bytes))

    assert res.status_code == 422
    assert res.json()["code"] == ErrorCode.NO_SPEECH_DETECTED.value


def test_verify_uses_best_of_multiple_enrollments(client, fresh_storage, monkeypatch):
    """여러 등록 발화가 있으면 가장 잘 맞는 것으로 판정한다."""
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "enroll_replaces_existing", False)

    user = unique_user()
    first = _wav(synth_speech(3.0, f0=100.0, seed=1))
    second = _wav(synth_speech(3.0, f0=230.0, seed=2))
    client.post(ENROLL, data={"user_id": user}, files=_files(first))
    client.post(ENROLL, data={"user_id": user}, files=_files(second))

    res = client.post(VERIFY, data={"user_id": user}, files=_files(second))

    assert res.status_code == 200
    body = res.json()
    assert body["compared_enrollments"] == 2
    # 두 번째 등록과 동일한 오디오이므로 그쪽과 완전히 일치해야 한다
    assert body["raw_cosine"] == pytest.approx(1.0, abs=1e-4)


def test_verification_is_audited(client, fresh_storage, speech_wav_bytes):
    """검증 시도는 성공·실패 모두 감사 로그에 남는다 (03 오딧 트레일)."""
    user = unique_user()
    client.post(ENROLL, data={"user_id": user}, files=_files(speech_wav_bytes))

    client.post(VERIFY, data={"user_id": user}, files=_files(speech_wav_bytes))
    client.post(VERIFY, data={"user_id": unique_user("ghost")}, files=_files(speech_wav_bytes))

    logs = fresh_storage.logs
    assert len(logs) == 2

    ok = logs[0]
    assert ok.outcome == "verified"
    assert ok.is_verified is True
    assert ok.raw_cosine is not None
    assert ok.threshold == pytest.approx(get_settings().match_threshold)
    assert ok.elapsed_ms is not None
    # 이 테스트는 코호트를 적재하지 않으므로 원시 코사인 폴백 경로다.
    # AS-Norm이 적용된 경로의 로그는 test_verify_asnorm.py가 검증한다.
    assert ok.normalized_score is None

    missing = logs[1]
    assert missing.outcome == "not_enrolled"
    assert missing.error_code == ErrorCode.NOT_ENROLLED.value


def test_health_reports_storage_backend(client, fresh_storage):
    """헬스 체크가 저장소 백엔드를 드러낸다 — 인메모리로 뜬 걸 모르고 지나치면 안 된다."""
    body = client.get("/api/v1/health").json()

    assert body["storage"] == "memory"
    assert body["storage_ok"] is True


def test_verify_response_is_extensible(client, fresh_storage, speech_wav_bytes):
    """Phase B~D 필드(normalized_score, spoof_score) 추가에 대비한 구조 (01 §2)."""
    user = unique_user()
    client.post(ENROLL, data={"user_id": user}, files=_files(speech_wav_bytes))

    body = client.post(VERIFY, data={"user_id": user}, files=_files(speech_wav_bytes)).json()

    assert set(body) >= {
        "status",
        "user_id",
        "is_verified",
        "match_probability",
        "raw_cosine",
        "threshold",
        "audio",
    }
    assert isinstance(body["audio"], dict)
