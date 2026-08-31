"""AS-Norm이 붙은 검증 경로 통합 테스트."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from .conftest import SAMPLE_RATE, synth_speech, unique_user

ENROLL = "/api/v1/enroll"
VERIFY = "/api/v1/verify"


def _wav(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _files(data: bytes) -> dict:
    return {"file": ("a.wav", data, "audio/wav")}


@pytest.fixture
def with_cohort(fresh_storage):
    """무작위 임포스터 코호트를 적재한 상태.

    실제 화자 데이터는 `eval/` 하네스가 다루고, 여기서는 AS-Norm 경로가 실제로
    타는지와 응답 계약이 맞는지를 본다.
    """
    from app.db import session as db_session
    from app.services.asnorm import CohortIndex

    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(120, 192)).astype(np.float32)
    index = CohortIndex(matrix, model="speechbrain/spkrec-ecapa-voxceleb", top_k=40)

    original = db_session._cohort
    db_session._cohort = index
    try:
        yield index
    finally:
        db_session._cohort = original


def test_verify_uses_asnorm_when_cohort_loaded(client, with_cohort, speech_wav_bytes):
    """코호트가 있으면 AS-Norm으로 판정하고 두 점수를 모두 반환한다 (FR-09)."""
    user = unique_user()
    client.post(ENROLL, data={"user_id": user}, files=_files(speech_wav_bytes))

    res = client.post(VERIFY, data={"user_id": user}, files=_files(speech_wav_bytes))

    assert res.status_code == 200
    body = res.json()
    assert body["scoring_method"] == "as_norm"
    assert body["normalized_score"] is not None
    assert body["raw_cosine"] == pytest.approx(1.0, abs=1e-4)
    # 정규화 점수는 코호트 표준편차 단위이므로 코사인의 [-1,1]에 갇히지 않는다
    assert abs(body["normalized_score"]) > 1.0
    assert body["is_verified"] is True


def test_verify_falls_back_to_raw_cosine_without_cohort(
    client, fresh_storage, speech_wav_bytes
):
    """코호트가 없으면 원시 코사인으로 폴백하고 그 사실을 드러낸다.

    정규화가 꺼진 채 운영되는 것을 모르고 지나치면 안 된다.
    """
    from app.db import session as db_session

    original = db_session._cohort
    db_session._cohort = None
    try:
        user = unique_user()
        client.post(ENROLL, data={"user_id": user}, files=_files(speech_wav_bytes))
        body = client.post(
            VERIFY, data={"user_id": user}, files=_files(speech_wav_bytes)
        ).json()
    finally:
        db_session._cohort = original

    assert body["scoring_method"] == "raw_cosine"
    assert body["normalized_score"] is None
    assert body["is_verified"] is True


def test_health_reports_asnorm_state(client, with_cohort):
    """헬스 체크가 AS-Norm 활성 여부와 코호트 크기를 드러낸다."""
    body = client.get("/api/v1/health").json()

    assert body["asnorm_active"] is True
    assert body["cohort_size"] == 120


def test_health_reports_asnorm_inactive_without_cohort(client, fresh_storage):
    from app.db import session as db_session

    original = db_session._cohort
    db_session._cohort = None
    try:
        body = client.get("/api/v1/health").json()
    finally:
        db_session._cohort = original

    assert body["asnorm_active"] is False
    assert body["cohort_size"] == 0


def test_asnorm_audit_log_records_normalized_score(client, with_cohort, speech_wav_bytes):
    """AS-Norm 적용 시 감사 로그에 정규화 점수가 남는다 (FR-09).

    원시 점수와 정규화 점수를 함께 남겨야 정규화 효과를 사후에 추적할 수 있다.
    """
    user = unique_user()
    client.post(ENROLL, data={"user_id": user}, files=_files(speech_wav_bytes))
    client.post(VERIFY, data={"user_id": user}, files=_files(speech_wav_bytes))

    log = with_cohort and client and None  # 가독성용 no-op
    from app.db import session as db_session

    logs = db_session.get_repository().logs
    assert len(logs) == 1
    assert logs[0].raw_cosine is not None
    assert logs[0].normalized_score is not None
    assert logs[0].outcome == "verified"


def test_asnorm_preserves_score_ordering(client, with_cohort):
    """정규화가 genuine/impostor의 순서를 뒤집지 않는다.

    AS-Norm은 점수 척도를 바꾸는 것이지 판정 방향을 바꾸는 것이 아니다.
    """
    user = unique_user()
    same = _wav(synth_speech(3.0, f0=110.0, seed=1))
    other = _wav(synth_speech(3.0, f0=240.0, seed=2))

    client.post(ENROLL, data={"user_id": user}, files=_files(same))
    a = client.post(VERIFY, data={"user_id": user}, files=_files(same)).json()
    b = client.post(VERIFY, data={"user_id": user}, files=_files(other)).json()

    assert a["raw_cosine"] > b["raw_cosine"]
    assert a["normalized_score"] > b["normalized_score"]


def test_cohort_seeding_and_loading_roundtrip(fresh_storage):
    """코호트를 저장소에 넣고 다시 읽어 인덱스를 만드는 경로."""
    import asyncio

    from app.services.asnorm import CohortIndex

    rng = np.random.default_rng(5)
    entries = [
        (rng.normal(size=192).astype(np.float32).tolist(), "test-model", f"spk{i}")
        for i in range(30)
    ]

    async def _run():
        inserted = await fresh_storage.add_cohort_entries(entries, source="unit-test")
        loaded = await fresh_storage.load_cohort("test-model")
        other = await fresh_storage.load_cohort("another-model")
        return inserted, loaded, other

    inserted, loaded, other = asyncio.get_event_loop().run_until_complete(_run())

    assert inserted == 30
    assert len(loaded) == 30
    assert other == []  # 모델별로 분리된다

    index = CohortIndex(
        np.asarray([e.embedding for e in loaded], dtype=np.float32), model="test-model", top_k=10
    )
    assert index.size == 30
    assert index.dim == 192
