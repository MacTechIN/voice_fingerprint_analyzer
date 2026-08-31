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
_cohort = None  # Optional[CohortIndex]


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
    global _pool, _repository, _backend, _cohort
    if _pool is not None:
        await _pool.close()
        _pool = None
    _repository = None
    _cohort = None
    _backend = "uninitialized"


def get_repository() -> SpeakerRepository:
    """현재 저장소. FastAPI 의존성으로 주입된다."""
    if _repository is None:
        raise RuntimeError("저장소가 초기화되지 않았습니다 (lifespan 미실행)")
    return _repository


def backend_name() -> str:
    """현재 저장소 백엔드 이름 — `/health`에 노출한다."""
    return _backend


async def load_cohort(model: str, top_k: int):
    """AS-Norm 임포스터 코호트를 메모리에 적재한다 (기동 시 1회).

    코호트는 고정 크기이고 요청마다 전부 참조되므로, 매 검증마다 DB에서 읽는
    대신 한 번 적재해 행렬 곱으로 처리한다.

    코호트가 비어 있으면 None을 반환하고 호출부가 원시 코사인으로 폴백한다.
    """
    global _cohort
    import numpy as np

    from app.services.asnorm import CohortIndex

    repo = get_repository()
    entries = await repo.load_cohort(model)
    if not entries:
        logger.warning(
            "모델 %s의 임포스터 코호트가 비어 있습니다. AS-Norm 없이 원시 코사인으로 "
            "판정합니다 — 점수 편차 보정이 적용되지 않습니다.",
            model,
        )
        _cohort = None
        return None

    matrix = np.asarray([e.embedding for e in entries], dtype=np.float32)
    _cohort = CohortIndex(matrix, model=model, top_k=top_k)
    logger.info("AS-Norm 코호트 적재 완료: %d개 (top_k=%d)", _cohort.size, top_k)
    return _cohort


def get_cohort():
    """적재된 코호트. 없으면 None (AS-Norm 비활성)."""
    return _cohort
