"""관리자 API 테스트.

인증과 집계 정확성을 확인한다. 감사 로그·사용자 ID가 노출되는 경로이므로
**인증이 실제로 막는지**가 가장 중요하다.
"""

from __future__ import annotations

import os

import pytest

TEST_DB_URL = os.environ.get("VG_TEST_DATABASE_URL", "")
ADMIN = "/api/v1/admin"
TOKEN = "test-admin-token"


class TestAdminAuth:
    """인증 — DB 없이도 검증할 수 있는 부분."""

    def test_no_token_configured_refuses_service(self, client, fresh_storage):
        """토큰 미설정 시 열어두지 않고 503으로 막는다.

        설정을 빠뜨린 배포가 조용히 공개되는 것이 가장 위험하다.
        """
        res = client.get(f"{ADMIN}/overview")

        assert res.status_code == 503
        assert "VG_ADMIN_TOKEN" in res.json()["detail"]

    def test_wrong_token_is_rejected(self, client, fresh_storage, monkeypatch):
        from app.config import get_settings

        monkeypatch.setattr(get_settings(), "admin_token", TOKEN)

        res = client.get(
            f"{ADMIN}/overview", headers={"Authorization": "Bearer wrong-token"}
        )

        assert res.status_code == 401

    def test_missing_bearer_prefix_is_rejected(self, client, fresh_storage, monkeypatch):
        from app.config import get_settings

        monkeypatch.setattr(get_settings(), "admin_token", TOKEN)

        res = client.get(f"{ADMIN}/overview", headers={"Authorization": TOKEN})

        assert res.status_code == 401

    def test_correct_token_passes_auth(self, client, fresh_storage, monkeypatch):
        """인증은 통과하되, 인메모리 저장소이므로 503으로 막힌다.

        401이 아니라 503이 나온다는 것이 인증을 통과했다는 증거다.
        """
        from app.config import get_settings

        monkeypatch.setattr(get_settings(), "admin_token", TOKEN)

        res = client.get(f"{ADMIN}/overview", headers={"Authorization": f"Bearer {TOKEN}"})

        assert res.status_code == 503
        assert "PostgreSQL" in res.json()["detail"]

    def test_all_admin_endpoints_require_auth(self, client, fresh_storage, monkeypatch):
        """엔드포인트를 추가하면서 인증 의존성을 빠뜨리는 것을 막는다."""
        from app.config import get_settings

        monkeypatch.setattr(get_settings(), "admin_token", TOKEN)

        paths = [
            "/overview",
            "/timeseries",
            "/score-distribution",
            "/threshold-impact?thresholds=1",
            "/speakers",
            "/attempts",
            "/calibration",
        ]
        for path in paths:
            res = client.get(f"{ADMIN}{path}")
            assert res.status_code == 401, f"{path} 가 인증 없이 접근되었다"


@pytest.mark.skipif(not TEST_DB_URL, reason="VG_TEST_DATABASE_URL 미설정")
class TestAnalyticsQueries:
    """집계 질의 — 실제 Postgres 대상."""

    @pytest.fixture
    async def analytics(self):
        import asyncpg

        from app.db.analytics import Analytics

        pool = await asyncpg.create_pool(TEST_DB_URL, min_size=1, max_size=2)
        # 이 테스트만의 데이터를 넣고 끝나면 지운다 — 다른 데이터를 건드리지 않는다
        await pool.execute(
            "DELETE FROM verification_attempts WHERE user_id LIKE 'analytics-test-%'"
        )
        try:
            yield Analytics(pool), pool
        finally:
            await pool.execute(
                "DELETE FROM verification_attempts WHERE user_id LIKE 'analytics-test-%'"
            )
            await pool.close()

    async def _seed(self, pool, rows: list[tuple]) -> None:
        """(outcome, is_verified, normalized_score, elapsed_ms) 튜플을 넣는다."""
        await pool.executemany(
            """
            INSERT INTO verification_attempts
                (user_id, outcome, is_verified, normalized_score, raw_cosine,
                 threshold, elapsed_ms)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            [
                ("analytics-test-user", outcome, verified, score, 0.5, 3.0, ms)
                for outcome, verified, score, ms in rows
            ],
        )

    async def test_overview_counts_outcomes(self, analytics):
        svc, pool = analytics
        await self._seed(
            pool,
            [
                ("verified", True, 9.0, 100.0),
                ("verified", True, 8.0, 200.0),
                ("rejected", False, -2.0, 150.0),
                ("audio_rejected", None, None, 50.0),
            ],
        )

        data = await svc.overview()

        by_outcome = {o.outcome: o.count for o in data.outcomes}
        assert by_outcome["verified"] >= 2
        assert by_outcome["rejected"] >= 1
        assert data.latency_avg_ms is not None
        assert data.latency_p95_ms is not None

    async def test_verified_ratio_excludes_non_decisions(self, analytics):
        """반려·미등록은 판정이 아니므로 통과율 계산에서 빠진다."""
        from app.db.analytics import Overview

        overview = Overview(
            total_attempts=10,
            outcomes=[],
            verified_count=3,
            rejected_count=1,
            audio_rejected_count=6,
            latency_avg_ms=None,
            latency_p50_ms=None,
            latency_p95_ms=None,
            total_enrollments=0,
            active_enrollments=0,
            enrolled_users=0,
            cohort_size=0,
            first_attempt_at=None,
            last_attempt_at=None,
        )

        assert overview.decision_count == 4
        assert overview.verified_ratio == pytest.approx(0.75)

    async def test_score_distribution_splits_by_verdict(self, analytics):
        svc, pool = analytics
        await self._seed(
            pool,
            [
                ("verified", True, 9.0, 100.0),
                ("verified", True, 9.5, 100.0),
                ("rejected", False, -2.0, 100.0),
            ],
        )

        dist = await svc.score_distribution(use_normalized=True, bins=8)

        assert dist.field_name == "normalized_score"
        assert dist.sample_count >= 3
        # 모든 표본이 히스토그램 어딘가에 들어가야 한다 — 상한 값이 누락되면 안 된다
        assert sum(b.total for b in dist.bins) == dist.sample_count

    async def test_threshold_impact_counts_flips(self, analytics):
        svc, pool = analytics
        await self._seed(
            pool,
            [
                ("verified", True, 9.0, 100.0),
                ("rejected", False, -2.0, 100.0),
            ],
        )

        # 임계값을 아주 높이면 통과했던 것이 거부로 뒤집힌다
        impacts = await svc.threshold_impact([100.0], use_normalized=True)

        assert impacts[0].would_pass == 0
        assert impacts[0].flipped_from_pass >= 1

    async def test_attempts_pagination(self, analytics):
        svc, pool = analytics
        await self._seed(pool, [("verified", True, 9.0, 100.0)] * 5)

        page = await svc.attempts(limit=2, offset=0, user_id="analytics-test-user")

        assert len(page.rows) == 2
        assert page.total >= 5
        assert page.has_more is True
        # 최신순 정렬 — 오딧 트레일은 최근 것부터 봐야 한다
        assert page.rows[0].id > page.rows[1].id

    async def test_attempts_filters_by_user(self, analytics):
        svc, pool = analytics
        await self._seed(pool, [("verified", True, 9.0, 100.0)])

        page = await svc.attempts(limit=50, user_id="nonexistent-user-xyz")

        assert page.total == 0
        assert page.rows == []

    async def test_speakers_flags_model_mismatch(self, analytics):
        """현재 모델과 다른 모델로 등록된 화자를 재등록 대상으로 표시한다."""
        svc, pool = analytics
        await pool.execute(
            """
            INSERT INTO speaker_enrollments (user_id, embedding, model, dim, l2_normalized)
            VALUES ('analytics-test-speaker', $1::vector, 'old/model-v1', 192, true)
            """,
            "[" + ",".join(["0.1"] * 192) + "]",
        )
        try:
            rows = await svc.speakers(current_model="new/model-v2")
            target = next(r for r in rows if r.user_id == "analytics-test-speaker")

            assert target.needs_reenrollment is True
            assert target.model == "old/model-v1"
            assert target.dim == 192
        finally:
            await pool.execute(
                "DELETE FROM speaker_enrollments WHERE user_id = 'analytics-test-speaker'"
            )
