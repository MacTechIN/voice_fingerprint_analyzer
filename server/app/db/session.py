"""저장소 수명 주기 관리.

`VG_DATABASE_URL`이 있으면 Postgres, 없으면 인메모리로 뜬다. DB 없이도 서버가
동작해야 개발·데모가 막히지 않지만, 인메모리로 뜬 사실은 로그와 `/health`에
분명히 드러내 프로덕션에서 모르고 지나치는 일이 없게 한다.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.db.repository import InMemoryRepository, PostgresRepository, SpeakerRepository

logger = logging.getLogger(__name__)

_repository: Optional[SpeakerRepository] = None
_pool = None
_backend: str = "uninitialized"


async def init(database_url: Optional[str]) -> SpeakerRepository:
    """저장소를 초기화한다. 기동 시 1회 호출."""
    global _repository, _pool, _backend

    if not database_url:
        logger.warning(
            "VG_DATABASE_URL이 설정되지 않아 인메모리 저장소로 기동합니다. "
            "프로세스를 재시작하면 등록된 성문이 모두 사라집니다."
        )
        _repository = InMemoryRepository()
        _backend = "memory"
        return _repository

    import asyncpg

    logger.info("PostgreSQL 연결 풀 생성 중")
    _pool = await asyncpg.create_pool(database_url, min_size=1, max_size=10)
    _repository = PostgresRepository(_pool)
    _backend = "postgres"
    logger.info("PostgreSQL 저장소 준비 완료")
    return _repository


async def close() -> None:
    """연결 풀을 정리한다. 종료 시 1회 호출."""
    global _pool, _repository, _backend
    if _pool is not None:
        await _pool.close()
        _pool = None
    _repository = None
    _backend = "uninitialized"


def get_repository() -> SpeakerRepository:
    """현재 저장소. FastAPI 의존성으로 주입된다."""
    if _repository is None:
        raise RuntimeError("저장소가 초기화되지 않았습니다 (lifespan 미실행)")
    return _repository


def backend_name() -> str:
    """현재 저장소 백엔드 이름 — `/health`에 노출한다."""
    return _backend
