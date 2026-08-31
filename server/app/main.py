"""FastAPI 애플리케이션 진입점.

실행:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.admin import router as admin_router
from app.api.routes import router
from app.config import get_settings
from app.core.errors import AudioRejected
from app.db import session as db_session
from app.schemas import ErrorResponse
from app.services import antispoof as antispoof_svc
from app.services import embedding as embedding_svc
from app.services import enhance as enhance_svc
from app.services import separation as separation_svc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """기동 시 저장소를 열고 모델을 미리 적재한다.

    모델 적재를 첫 요청까지 미루면 그 요청만 수십 초를 기다리게 된다. 헬스 체크가
    통과한 시점에는 실제로 응답할 준비가 되어 있어야 한다.
    """
    settings = get_settings()

    # PyTorch 스레드 수를 기동 시 한 번 고정한다. 추론이 워커 스레드에서 도는데
    # 그 안에서 OpenMP 병렬 영역이 확장되지 않아 사실상 1스레드로 떨어지는
    # 경우가 있고, 그러면 분리 추론이 4배 느려진다 (config 주석 참조).
    if settings.torch_num_threads > 0:
        import torch

        torch.set_num_threads(settings.torch_num_threads)
        logger.info("PyTorch 스레드 수 고정: %d", settings.torch_num_threads)

    # 저장소는 실패하면 기동을 중단한다. 모델 워밍업과 달리 저장소 없이는
    # 등록·검증이 아예 성립하지 않으므로, 반쯤 동작하는 서버를 띄우는 것보다
    # 즉시 실패하는 편이 낫다.
    await db_session.init(settings.database_url or None)

    # AS-Norm 코호트 적재. 비어 있으면 원시 코사인으로 폴백하며, 그 사실은
    # 경고 로그와 /health의 asnorm_active에 드러난다.
    if settings.asnorm_enabled:
        await db_session.load_cohort(settings.embedding_model, settings.asnorm_top_k)

    if settings.warmup_on_startup:
        logger.info("모델 워밍업 시작")
        if settings.antispoof_enabled:
            try:
                antispoof_svc.warmup(settings.antispoof_weights)
            except Exception:
                logger.exception(
                    "딥페이크 탐지 모델 적재 실패 — 가중치 경로를 확인하세요: %s",
                    settings.antispoof_weights,
                )
        if settings.separation_enabled:
            try:
                separation_svc.warmup(settings.separation_model, settings.model_cache_dir)
            except Exception:
                logger.exception("음성 분리 모델 적재 실패 — 첫 요청 시 재시도한다")
        if settings.enhance_enabled:
            try:
                enhance_svc.warmup()
            except Exception:
                logger.exception("음성 향상 모델 적재 실패 — 첫 요청 시 재시도한다")
        try:
            embedding_svc.warmup(
                settings.embedding_model,
                settings.model_cache_dir,
                backend=settings.embedding_backend,
                onnx_threads=settings.onnx_intra_op_threads,
            )
            logger.info("모델 워밍업 완료")
        except Exception:
            # 워밍업 실패로 서버를 죽이지는 않는다. 첫 요청에서 재시도되며,
            # 그때도 실패하면 500으로 드러난다.
            logger.exception("모델 워밍업 실패 — 첫 요청 시 재시도한다")

    try:
        yield
    finally:
        await db_session.close()


app = FastAPI(
    title="VoiceGuard Verification API",
    description=(
        "서버 집중형 화자 인증(성문 분석) API. "
        "Phase 6(캘리브레이션) 범위: VAD 전처리 + ECAPA-TDNN 임베딩, 성문 등록·1:1 검증, "
        "AS-Norm 점수 정규화 및 EER 기반 임계값."
    ),
    version="0.3.0",
    lifespan=lifespan,
)


@app.exception_handler(AudioRejected)
async def audio_rejected_handler(request: Request, exc: AudioRejected) -> JSONResponse:
    """오디오 반려를 422로 변환한다.

    서버 장애가 아니라 입력 문제이므로 5xx가 아니다. 클라이언트는 `code`로
    재녹음 안내를 분기한다.
    """
    logger.info("오디오 반려: code=%s detail=%s context=%s", exc.code, exc.detail, exc.context)
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(code=exc.code.value, detail=exc.detail).model_dump(),
    )


app.include_router(router, prefix=API_PREFIX)
app.include_router(admin_router, prefix=API_PREFIX)


@app.get("/", tags=["ops"], include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": "voiceguard-verification", "docs": "/docs", "api": API_PREFIX}
