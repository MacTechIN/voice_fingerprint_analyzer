"""저장소 계약 테스트.

인메모리와 Postgres 두 구현이 같은 계약을 지키는지 동일한 테스트로 확인한다.
Postgres는 `VG_TEST_DATABASE_URL`이 있을 때만 돈다 — DB 없는 환경에서 테스트
전체가 실패하면 안 되기 때문이다.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio  # noqa: F401  (설치되어 있지 않으면 아래에서 건너뛴다)

from app.db.repository import InMemoryRepository, VerificationLog

TEST_DB_URL = os.environ.get("VG_TEST_DATABASE_URL", "")

# 192차원을 요구하는 것은 pgvector 스키마의 제약이다. 인메모리 구현에도 같은
# 크기를 써서 두 구현을 정확히 같은 조건으로 비교한다.
DIM = 192


def _vec(seed: float) -> list[float]:
    return [seed] + [0.0] * (DIM - 1)


async def _exercise_contract(repo) -> None:
    """저장소 구현이 지켜야 할 계약."""
    user = "contract-user"

    # 미등록 사용자는 빈 리스트
    assert await repo.list_active_enrollments(user) == []

    saved = await repo.add_enrollment(
        user_id=user,
        embedding=_vec(1.0),
        model="test-model",
        dim=DIM,
        l2_normalized=True,
        speech_duration_sec=3.5,
    )
    assert saved.id > 0
    assert saved.user_id == user
    assert saved.model == "test-model"
    assert saved.dim == DIM
    assert saved.l2_normalized is True
    assert saved.speech_duration_sec == pytest.approx(3.5)
    assert len(saved.embedding) == DIM
    assert saved.embedding[0] == pytest.approx(1.0)

    # 조회로 되읽었을 때 벡터가 보존되어야 한다 — 직렬화 왕복이 손실 없이 돌아야
    # 저장된 성문으로 검증할 수 있다
    rows = await repo.list_active_enrollments(user)
    assert len(rows) == 1
    assert rows[0].embedding[0] == pytest.approx(1.0)

    # 두 번째 등록 후 비활성화
    await repo.add_enrollment(
        user_id=user,
        embedding=_vec(0.5),
        model="test-model",
        dim=DIM,
        l2_normalized=True,
        speech_duration_sec=None,
    )
    assert len(await repo.list_active_enrollments(user)) == 2

    deactivated = await repo.deactivate_enrollments(user)
    assert deactivated == 2
    assert await repo.list_active_enrollments(user) == []

    # 다른 사용자의 성문은 섞이지 않는다
    await repo.add_enrollment(
        user_id="other-user",
        embedding=_vec(0.25),
        model="test-model",
        dim=DIM,
        l2_normalized=True,
        speech_duration_sec=None,
    )
    assert await repo.list_active_enrollments(user) == []
    assert len(await repo.list_active_enrollments("other-user")) == 1

    # 감사 로그
    await repo.log_verification(
        VerificationLog(
            user_id=user,
            outcome="verified",
            raw_cosine=0.87,
            match_probability=91.3,
            is_verified=True,
            threshold=0.25,
            model="test-model",
            elapsed_ms=123.4,
        )
    )
    assert await repo.health() is True


@pytest.mark.asyncio
async def test_in_memory_repository_contract():
    await _exercise_contract(InMemoryRepository())


@pytest.mark.skipif(not TEST_DB_URL, reason="VG_TEST_DATABASE_URL 미설정")
@pytest.mark.asyncio
async def test_postgres_repository_contract():
    """실제 pgvector에 대해 같은 계약을 검증한다."""
    import asyncpg

    from app.db.repository import PostgresRepository

    pool = await asyncpg.create_pool(TEST_DB_URL, min_size=1, max_size=2)
    try:
        # 계약 테스트가 쓰는 사용자만 정리한다 — 다른 데이터를 지우지 않는다
        await pool.execute(
            "DELETE FROM speaker_enrollments WHERE user_id IN ('contract-user','other-user')"
        )
        await pool.execute(
            "DELETE FROM verification_attempts WHERE user_id IN ('contract-user','other-user')"
        )
        await _exercise_contract(PostgresRepository(pool))
    finally:
        await pool.execute(
            "DELETE FROM speaker_enrollments WHERE user_id IN ('contract-user','other-user')"
        )
        await pool.execute(
            "DELETE FROM verification_attempts WHERE user_id IN ('contract-user','other-user')"
        )
        await pool.close()


@pytest.mark.skipif(not TEST_DB_URL, reason="VG_TEST_DATABASE_URL 미설정")
@pytest.mark.asyncio
async def test_postgres_preserves_full_vector_precision():
    """192개 성분이 왕복에서 전부 보존되는지 확인한다.

    pgvector는 float4로 저장하므로 float64 원본과 완전히 같지는 않다. 허용 오차를
    명시해 두어, 나중에 정밀도가 더 나빠지면 테스트가 잡아내게 한다.
    """
    import asyncpg

    from app.db.repository import PostgresRepository

    original = [((i * 37) % 100) / 100.0 - 0.5 for i in range(DIM)]
    pool = await asyncpg.create_pool(TEST_DB_URL, min_size=1, max_size=2)
    try:
        repo = PostgresRepository(pool)
        saved = await repo.add_enrollment(
            user_id="precision-user",
            embedding=original,
            model="test-model",
            dim=DIM,
            l2_normalized=False,
            speech_duration_sec=None,
        )
        assert len(saved.embedding) == DIM
        for a, b in zip(original, saved.embedding):
            assert a == pytest.approx(b, abs=1e-6)
    finally:
        await pool.execute("DELETE FROM speaker_enrollments WHERE user_id = 'precision-user'")
        await pool.close()
