"""관리자 대시보드용 집계 질의.

대시보드가 DB에 직접 붙는 대신 서버를 거치게 한 이유: 스키마 지식을 한 곳에
모아두기 위해서다. 웹 계층이 테이블 구조를 알면 마이그레이션할 때마다 두 곳을
고쳐야 하고, DB 자격증명을 웹 티어에도 두어야 한다.

**중요한 한계 — 운영 로그로는 FAR/FRR을 계산할 수 없다.**
감사 로그에는 판정 결과(`is_verified`)만 있고 **정답 레이블이 없다.** 검증을
시도한 사람이 실제로 본인이었는지 서버는 알 수 없기 때문이다. 따라서
오수락률·오거부률은 레이블이 있는 오프라인 평가(`eval/`)에서만 나온다.
운영 데이터로는 **점수 분포**와 **임계값 변경 시 판정이 뒤집히는 건수**까지만
말할 수 있으며, 이 모듈은 그 선을 넘지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class OutcomeCount:
    outcome: str
    count: int


@dataclass(frozen=True)
class Overview:
    """대시보드 상단 요약 (03 §3)."""

    total_attempts: int
    outcomes: list[OutcomeCount]
    verified_count: int
    rejected_count: int
    audio_rejected_count: int

    latency_avg_ms: Optional[float]
    latency_p50_ms: Optional[float]
    latency_p95_ms: Optional[float]

    total_enrollments: int
    active_enrollments: int
    enrolled_users: int
    cohort_size: int

    first_attempt_at: Optional[datetime]
    last_attempt_at: Optional[datetime]

    @property
    def decision_count(self) -> int:
        """실제로 판정까지 간 시도 수 (반려·미등록 제외)."""
        return self.verified_count + self.rejected_count

    @property
    def verified_ratio(self) -> Optional[float]:
        """판정 중 통과 비율.

        **정확도가 아니다.** 시도한 사람이 본인이었는지 알 수 없으므로, 이 값이
        높다고 시스템이 정확한 것도 낮다고 부정확한 것도 아니다. 사용 패턴 지표다.
        """
        if self.decision_count == 0:
            return None
        return self.verified_count / self.decision_count


@dataclass(frozen=True)
class TimeBucket:
    bucket: datetime
    verified: int
    rejected: int
    audio_rejected: int
    not_enrolled: int
    other: int
    avg_elapsed_ms: Optional[float]

    @property
    def total(self) -> int:
        return (
            self.verified
            + self.rejected
            + self.audio_rejected
            + self.not_enrolled
            + self.other
        )


@dataclass(frozen=True)
class ScoreBin:
    lower: float
    upper: float
    verified: int
    rejected: int

    @property
    def total(self) -> int:
        return self.verified + self.rejected


@dataclass(frozen=True)
class ScoreDistribution:
    """판정 점수 분포.

    genuine/impostor로 나눌 수 없다 — 레이블이 없기 때문이다. 대신 판정 결과로
    나눈다. 분포가 뚜렷이 갈리면 임계값이 분리 지점에 잘 놓였다는 방증이지만,
    그것이 정확도를 증명하지는 않는다.
    """

    field_name: str
    bins: list[ScoreBin]
    threshold: Optional[float]
    sample_count: int
    min_score: Optional[float]
    max_score: Optional[float]


@dataclass(frozen=True)
class ThresholdImpact:
    """임계값을 바꿨을 때 과거 시도의 판정이 어떻게 달라지는가.

    **오류율이 아니라 영향도다.** 통과가 거부로 바뀐다는 것은 알 수 있어도,
    그 변경이 옳은지는 레이블 없이 말할 수 없다.
    """

    threshold: float
    would_pass: int
    would_fail: int
    flipped_from_pass: int
    flipped_from_fail: int

    @property
    def total(self) -> int:
        return self.would_pass + self.would_fail


@dataclass(frozen=True)
class SpeakerRow:
    """화자 데이터베이스 한 행 (03 §3)."""

    user_id: str
    active_count: int
    inactive_count: int
    model: Optional[str]
    dim: Optional[int]
    last_enrolled_at: Optional[datetime]
    avg_speech_duration_sec: Optional[float]
    needs_reenrollment: bool
    """현재 임베딩 모델과 다른 모델로 등록됨 — 검증이 model_mismatch로 거부된다."""


@dataclass(frozen=True)
class AttemptRow:
    """오딧 트레일 한 행 (03 §3)."""

    id: int
    user_id: str
    outcome: str
    is_verified: Optional[bool]
    raw_cosine: Optional[float]
    normalized_score: Optional[float]
    match_probability: Optional[float]
    threshold: Optional[float]
    model: Optional[str]
    speech_duration_sec: Optional[float]
    error_code: Optional[str]
    client_ip: Optional[str]
    elapsed_ms: Optional[float]
    created_at: datetime


@dataclass(frozen=True)
class AttemptPage:
    rows: list[AttemptRow] = field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.rows) < self.total


class Analytics:
    """집계 질의 모음. PostgreSQL 전용이다.

    인메모리 저장소로 뜬 서버에서는 대시보드를 쓸 수 없다 — 통계를 볼 만큼
    데이터가 쌓이는 배포라면 DB가 있어야 한다.
    """

    def __init__(self, pool) -> None:
        self._pool = pool

    async def overview(
        self, *, hours: Optional[int] = None, current_model: Optional[str] = None
    ) -> Overview:
        window = "WHERE created_at >= NOW() - ($1 || ' hours')::interval" if hours else ""
        args = [str(hours)] if hours else []

        rows = await self._pool.fetch(
            f"SELECT outcome, count(*) AS n FROM verification_attempts {window} GROUP BY outcome",
            *args,
        )
        outcomes = [OutcomeCount(outcome=r["outcome"], count=r["n"]) for r in rows]
        by_outcome = {o.outcome: o.count for o in outcomes}

        latency = await self._pool.fetchrow(
            f"""
            SELECT avg(elapsed_ms) AS avg_ms,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY elapsed_ms) AS p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY elapsed_ms) AS p95,
                   min(created_at) AS first_at,
                   max(created_at) AS last_at
            FROM verification_attempts {window}
            """,
            *args,
        )

        enrollments = await self._pool.fetchrow(
            """
            SELECT count(*) AS total,
                   count(*) FILTER (WHERE is_active) AS active,
                   count(DISTINCT user_id) FILTER (WHERE is_active) AS users
            FROM speaker_enrollments
            """
        )
        # 코호트는 모델별로 공존한다(백엔드를 바꾸면 새로 적재하고 옛것을 남겨둔다).
        # 실제로 정규화에 쓰이는 것은 현재 모델의 코호트뿐이므로 그것만 센다.
        if current_model:
            cohort = await self._pool.fetchval(
                "SELECT count(*) FROM impostor_cohort WHERE model = $1", current_model
            )
        else:
            cohort = await self._pool.fetchval("SELECT count(*) FROM impostor_cohort")

        return Overview(
            total_attempts=sum(by_outcome.values()),
            outcomes=sorted(outcomes, key=lambda o: -o.count),
            verified_count=by_outcome.get("verified", 0),
            rejected_count=by_outcome.get("rejected", 0),
            audio_rejected_count=by_outcome.get("audio_rejected", 0),
            latency_avg_ms=_f(latency["avg_ms"]),
            latency_p50_ms=_f(latency["p50"]),
            latency_p95_ms=_f(latency["p95"]),
            total_enrollments=enrollments["total"],
            active_enrollments=enrollments["active"],
            enrolled_users=enrollments["users"],
            cohort_size=cohort or 0,
            first_attempt_at=latency["first_at"],
            last_attempt_at=latency["last_at"],
        )

    async def timeseries(self, *, hours: int = 24, bucket_minutes: int = 60) -> list[TimeBucket]:
        rows = await self._pool.fetch(
            """
            SELECT
                to_timestamp(
                    floor(extract(epoch FROM created_at) / ($2 * 60)) * ($2 * 60)
                ) AS bucket,
                count(*) FILTER (WHERE outcome = 'verified') AS verified,
                count(*) FILTER (WHERE outcome = 'rejected') AS rejected,
                count(*) FILTER (WHERE outcome = 'audio_rejected') AS audio_rejected,
                count(*) FILTER (WHERE outcome = 'not_enrolled') AS not_enrolled,
                count(*) FILTER (
                    WHERE outcome NOT IN ('verified','rejected','audio_rejected','not_enrolled')
                ) AS other,
                avg(elapsed_ms) AS avg_ms
            FROM verification_attempts
            WHERE created_at >= NOW() - ($1 || ' hours')::interval
            GROUP BY bucket
            ORDER BY bucket
            """,
            str(hours),
            bucket_minutes,
        )
        return [
            TimeBucket(
                bucket=r["bucket"],
                verified=r["verified"],
                rejected=r["rejected"],
                audio_rejected=r["audio_rejected"],
                not_enrolled=r["not_enrolled"],
                other=r["other"],
                avg_elapsed_ms=_f(r["avg_ms"]),
            )
            for r in rows
        ]

    async def score_distribution(
        self, *, use_normalized: bool = True, bins: int = 20
    ) -> ScoreDistribution:
        """판정 점수 히스토그램.

        AS-Norm이 켜져 있으면 정규화 점수를, 아니면 원시 코사인을 본다. 둘은
        척도가 완전히 다르므로 같은 축에 섞으면 안 된다.
        """
        column = "normalized_score" if use_normalized else "raw_cosine"

        stats = await self._pool.fetchrow(
            f"""
            SELECT min({column}) AS lo, max({column}) AS hi, count(*) AS n,
                   max(threshold) AS thr
            FROM verification_attempts
            WHERE {column} IS NOT NULL AND outcome IN ('verified','rejected')
            """
        )
        count = stats["n"] or 0
        if count == 0:
            return ScoreDistribution(
                field_name=column, bins=[], threshold=None,
                sample_count=0, min_score=None, max_score=None,
            )

        lo, hi = float(stats["lo"]), float(stats["hi"])
        if hi <= lo:
            # 표본이 하나이거나 모두 같은 값이면 폭을 임의로 준다
            hi = lo + 1.0

        rows = await self._pool.fetch(
            f"""
            SELECT width_bucket({column}, $1, $2, $3) AS idx,
                   count(*) FILTER (WHERE is_verified) AS verified,
                   count(*) FILTER (WHERE NOT is_verified) AS rejected
            FROM verification_attempts
            WHERE {column} IS NOT NULL AND outcome IN ('verified','rejected')
            GROUP BY idx ORDER BY idx
            """,
            lo, hi, bins,
        )
        width = (hi - lo) / bins
        by_index = {r["idx"]: r for r in rows}

        histogram = []
        for i in range(1, bins + 1):
            r = by_index.get(i)
            # width_bucket은 상한과 같은 값을 bins+1로 보낸다. 마지막 칸에 합친다.
            if i == bins and (bins + 1) in by_index:
                overflow = by_index[bins + 1]
                verified = (r["verified"] if r else 0) + overflow["verified"]
                rejected = (r["rejected"] if r else 0) + overflow["rejected"]
            else:
                verified = r["verified"] if r else 0
                rejected = r["rejected"] if r else 0
            histogram.append(
                ScoreBin(
                    lower=lo + width * (i - 1),
                    upper=lo + width * i,
                    verified=verified,
                    rejected=rejected,
                )
            )

        return ScoreDistribution(
            field_name=column,
            bins=histogram,
            threshold=_f(stats["thr"]),
            sample_count=count,
            min_score=lo,
            max_score=float(stats["hi"]),
        )

    async def threshold_impact(
        self, candidates: list[float], *, use_normalized: bool = True
    ) -> list[ThresholdImpact]:
        """후보 임계값별로 과거 시도의 판정이 어떻게 달라지는지 센다."""
        column = "normalized_score" if use_normalized else "raw_cosine"
        rows = await self._pool.fetch(
            f"""
            SELECT {column} AS score, is_verified
            FROM verification_attempts
            WHERE {column} IS NOT NULL AND outcome IN ('verified','rejected')
            """
        )
        scores = [(float(r["score"]), bool(r["is_verified"])) for r in rows]

        impacts = []
        for threshold in candidates:
            would_pass = would_fail = flipped_pass = flipped_fail = 0
            for score, was_verified in scores:
                passes = score >= threshold
                if passes:
                    would_pass += 1
                    if not was_verified:
                        flipped_fail += 1  # 거부됐던 것이 통과로
                else:
                    would_fail += 1
                    if was_verified:
                        flipped_pass += 1  # 통과했던 것이 거부로
            impacts.append(
                ThresholdImpact(
                    threshold=threshold,
                    would_pass=would_pass,
                    would_fail=would_fail,
                    flipped_from_pass=flipped_pass,
                    flipped_from_fail=flipped_fail,
                )
            )
        return impacts

    async def speakers(self, *, current_model: str, limit: int = 200) -> list[SpeakerRow]:
        rows = await self._pool.fetch(
            """
            SELECT user_id,
                   count(*) FILTER (WHERE is_active) AS active_count,
                   count(*) FILTER (WHERE NOT is_active) AS inactive_count,
                   (array_agg(model ORDER BY created_at DESC)
                        FILTER (WHERE is_active))[1] AS model,
                   (array_agg(dim ORDER BY created_at DESC)
                        FILTER (WHERE is_active))[1] AS dim,
                   max(created_at) FILTER (WHERE is_active) AS last_enrolled_at,
                   avg(speech_duration_sec) FILTER (WHERE is_active) AS avg_speech
            FROM speaker_enrollments
            GROUP BY user_id
            ORDER BY max(created_at) DESC
            LIMIT $1
            """,
            limit,
        )
        return [
            SpeakerRow(
                user_id=r["user_id"],
                active_count=r["active_count"],
                inactive_count=r["inactive_count"],
                model=r["model"],
                dim=r["dim"],
                last_enrolled_at=r["last_enrolled_at"],
                avg_speech_duration_sec=_f(r["avg_speech"]),
                # 활성 등록이 있는데 모델이 다르면 검증이 거부된다 (02 §6)
                needs_reenrollment=bool(
                    r["active_count"] and r["model"] and r["model"] != current_model
                ),
            )
            for r in rows
        ]

    async def attempts(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        user_id: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> AttemptPage:
        conditions = []
        args: list = []
        if user_id:
            args.append(user_id)
            conditions.append(f"user_id = ${len(args)}")
        if outcome:
            args.append(outcome)
            conditions.append(f"outcome = ${len(args)}")
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        total = await self._pool.fetchval(
            f"SELECT count(*) FROM verification_attempts {where}", *args
        )
        rows = await self._pool.fetch(
            f"""
            SELECT id, user_id, outcome, is_verified, raw_cosine, normalized_score,
                   match_probability, threshold, model, speech_duration_sec,
                   error_code, client_ip, elapsed_ms, created_at
            FROM verification_attempts {where}
            ORDER BY id DESC LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}
            """,
            *args, limit, offset,
        )
        return AttemptPage(
            rows=[
                AttemptRow(
                    id=r["id"],
                    user_id=r["user_id"],
                    outcome=r["outcome"],
                    is_verified=r["is_verified"],
                    raw_cosine=_f(r["raw_cosine"]),
                    normalized_score=_f(r["normalized_score"]),
                    match_probability=_f(r["match_probability"]),
                    threshold=_f(r["threshold"]),
                    model=r["model"],
                    speech_duration_sec=_f(r["speech_duration_sec"]),
                    error_code=r["error_code"],
                    client_ip=r["client_ip"],
                    elapsed_ms=_f(r["elapsed_ms"]),
                    created_at=r["created_at"],
                )
                for r in rows
            ],
            total=total or 0,
            limit=limit,
            offset=offset,
        )


def _f(value) -> Optional[float]:
    return float(value) if value is not None else None
