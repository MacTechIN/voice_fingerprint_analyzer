"""성문 저장소 인터페이스와 두 가지 구현.

Postgres(pgvector)가 프로덕션 경로이고, 인메모리 구현은 DB 없이 개발·테스트할
때 쓴다. 두 구현이 같은 계약을 지키므로 라우트는 어느 쪽인지 알 필요가 없다.

pgvector를 쓰면서도 유사도 계산을 파이썬(app.services.scoring)에서 하는 이유:
1:1 검증은 사용자당 소수의 벡터만 비교하므로 SQL로 넘길 이득이 없고, Phase 6에
AS-Norm이 들어오면 어차피 스코어 후처리를 애플리케이션에서 해야 한다. 계산
위치를 한 곳에 모아두면 그때 흩어진 로직을 찾아다니지 않아도 된다.
1:N 식별을 도입하면 그때는 `<=>` 연산자와 HNSW 인덱스로 옮긴다.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class Enrollment:
    """저장된 등록 성문."""

    id: int
    user_id: str
    embedding: list[float]
    model: str
    dim: int
    l2_normalized: bool
    speech_duration_sec: Optional[float]
    created_at: datetime


@dataclass
class VerificationLog:
    """검증 시도 감사 로그 (03 오딧 트레일)."""

    user_id: str
    outcome: str
    raw_cosine: Optional[float] = None
    normalized_score: Optional[float] = None  # Phase 6 AS-Norm
    match_probability: Optional[float] = None
    is_verified: Optional[bool] = None
    threshold: Optional[float] = None
    model: Optional[str] = None
    speech_duration_sec: Optional[float] = None
    error_code: Optional[str] = None
    client_ip: Optional[str] = None
    elapsed_ms: Optional[float] = None


class SpeakerRepository(ABC):
    """성문 저장소 계약."""

    @abstractmethod
    async def add_enrollment(
        self,
        *,
        user_id: str,
        embedding: list[float],
        model: str,
        dim: int,
        l2_normalized: bool,
        speech_duration_sec: Optional[float],
    ) -> Enrollment:
        """등록 성문을 저장하고 저장된 행을 반환한다."""

    @abstractmethod
    async def list_active_enrollments(self, user_id: str) -> list[Enrollment]:
        """해당 사용자의 활성 등록 성문 목록. 없으면 빈 리스트."""

    @abstractmethod
    async def deactivate_enrollments(self, user_id: str) -> int:
        """사용자의 기존 등록을 비활성화하고 건수를 반환한다.

        삭제가 아니라 비활성화인 이유: 재등록 후 문제가 생겼을 때 이전 벡터를
        추적할 수 있어야 하고, 감사 로그가 참조하는 이력이 사라지면 안 된다.
        """

    @abstractmethod
    async def log_verification(self, log: VerificationLog) -> None:
        """검증 시도를 기록한다. 실패해도 검증 응답을 막지 않는다."""

    @abstractmethod
    async def health(self) -> bool:
        """저장소 접속 가능 여부."""


class InMemoryRepository(SpeakerRepository):
    """프로세스 메모리 저장소.

    DB 없이 서버를 띄우거나 테스트할 때 쓴다. 프로세스가 죽으면 사라지므로
    프로덕션 용도가 아니다.
    """

    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._logs: list[VerificationLog] = []
        self._next_id = 1

    async def add_enrollment(
        self,
        *,
        user_id: str,
        embedding: list[float],
        model: str,
        dim: int,
        l2_normalized: bool,
        speech_duration_sec: Optional[float],
    ) -> Enrollment:
        row = {
            "id": self._next_id,
            "user_id": user_id,
            "embedding": list(embedding),
            "model": model,
            "dim": dim,
            "l2_normalized": l2_normalized,
            "speech_duration_sec": speech_duration_sec,
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
        }
        self._next_id += 1
        self._rows.append(row)
        return self._to_enrollment(row)

    async def list_active_enrollments(self, user_id: str) -> list[Enrollment]:
        return [
            self._to_enrollment(r)
            for r in self._rows
            if r["user_id"] == user_id and r["is_active"]
        ]

    async def deactivate_enrollments(self, user_id: str) -> int:
        count = 0
        for r in self._rows:
            if r["user_id"] == user_id and r["is_active"]:
                r["is_active"] = False
                count += 1
        return count

    async def log_verification(self, log: VerificationLog) -> None:
        self._logs.append(log)

    async def health(self) -> bool:
        return True

    @property
    def logs(self) -> list[VerificationLog]:
        """테스트 검사용."""
        return self._logs

    @staticmethod
    def _to_enrollment(row: dict) -> Enrollment:
        return Enrollment(
            id=row["id"],
            user_id=row["user_id"],
            embedding=list(row["embedding"]),
            model=row["model"],
            dim=row["dim"],
            l2_normalized=row["l2_normalized"],
            speech_duration_sec=row["speech_duration_sec"],
            created_at=row["created_at"],
        )


def _to_pgvector(embedding: list[float]) -> str:
    """파이썬 리스트를 pgvector 리터럴로 변환한다.

    asyncpg는 vector 타입을 모르므로 문자열로 넘기고 SQL에서 캐스팅한다.
    """
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def _from_pgvector(value: str | list[float]) -> list[float]:
    """pgvector 값을 파이썬 리스트로 되돌린다."""
    if isinstance(value, list):
        return [float(x) for x in value]
    return [float(x) for x in json.loads(value)]


class PostgresRepository(SpeakerRepository):
    """PostgreSQL + pgvector 저장소 (Supabase 호환)."""

    def __init__(self, pool) -> None:
        self._pool = pool

    async def add_enrollment(
        self,
        *,
        user_id: str,
        embedding: list[float],
        model: str,
        dim: int,
        l2_normalized: bool,
        speech_duration_sec: Optional[float],
    ) -> Enrollment:
        row = await self._pool.fetchrow(
            """
            INSERT INTO speaker_enrollments
                (user_id, embedding, model, dim, l2_normalized, speech_duration_sec)
            VALUES ($1, $2::vector, $3, $4, $5, $6)
            RETURNING id, user_id, embedding::text AS embedding, model, dim,
                      l2_normalized, speech_duration_sec, created_at
            """,
            user_id,
            _to_pgvector(embedding),
            model,
            dim,
            l2_normalized,
            speech_duration_sec,
        )
        return self._to_enrollment(row)

    async def list_active_enrollments(self, user_id: str) -> list[Enrollment]:
        rows = await self._pool.fetch(
            """
            SELECT id, user_id, embedding::text AS embedding, model, dim,
                   l2_normalized, speech_duration_sec, created_at
            FROM speaker_enrollments
            WHERE user_id = $1 AND is_active
            ORDER BY created_at DESC
            """,
            user_id,
        )
        return [self._to_enrollment(r) for r in rows]

    async def deactivate_enrollments(self, user_id: str) -> int:
        result = await self._pool.execute(
            "UPDATE speaker_enrollments SET is_active = FALSE WHERE user_id = $1 AND is_active",
            user_id,
        )
        # asyncpg는 "UPDATE <n>" 형태의 상태 문자열을 준다
        return int(result.split()[-1]) if result else 0

    async def log_verification(self, log: VerificationLog) -> None:
        await self._pool.execute(
            """
            INSERT INTO verification_attempts
                (user_id, raw_cosine, normalized_score, match_probability, is_verified,
                 threshold, model, speech_duration_sec, outcome, error_code,
                 client_ip, elapsed_ms)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            """,
            log.user_id,
            log.raw_cosine,
            log.normalized_score,
            log.match_probability,
            log.is_verified,
            log.threshold,
            log.model,
            log.speech_duration_sec,
            log.outcome,
            log.error_code,
            log.client_ip,
            log.elapsed_ms,
        )

    async def health(self) -> bool:
        try:
            await self._pool.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    @staticmethod
    def _to_enrollment(row) -> Enrollment:
        return Enrollment(
            id=row["id"],
            user_id=row["user_id"],
            embedding=_from_pgvector(row["embedding"]),
            model=row["model"],
            dim=row["dim"],
            l2_normalized=row["l2_normalized"],
            speech_duration_sec=row["speech_duration_sec"],
            created_at=row["created_at"],
        )
