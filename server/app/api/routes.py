"""API 라우트.

Phase 1 범위는 `POST /api/v1/extract` 하나다. 등록(`/enroll`)과 검증(`/verify`)은
벡터 DB가 붙는 Phase 2에서 추가한다 (06 Phase 2).

경로에 버전을 박아두는 이유는 검증 API가 이후 기능 보강이 예정된 확장 지점이기
때문이다 (01 §2).
"""

from __future__ import annotations

import logging
import time

import anyio
from fastapi import APIRouter, Depends, File, UploadFile

from app.config import Settings, get_settings
from app.core.errors import AudioRejected, ErrorCode
from app.schemas import (
    AudioInfo,
    EmbeddingInfo,
    ExtractResponse,
    HealthResponse,
    SpeechSegmentOut,
)
from app.services import audio as audio_svc
from app.services import embedding as embedding_svc
from app.services import vad as vad_svc

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """서비스 생존 및 모델 적재 상태."""
    return HealthResponse(
        status="ok",
        model=settings.embedding_model,
        models_loaded=embedding_svc._classifier is not None,
    )


def _run_pipeline(raw: bytes, settings: Settings) -> ExtractResponse:
    """디코딩 → VAD → 임베딩. CPU 바운드이므로 워커 스레드에서 실행된다."""
    started = time.perf_counter()

    decoded = audio_svc.decode(
        raw,
        target_rate=settings.target_sample_rate,
        max_sec=settings.max_audio_sec,
    )
    vad_result = vad_svc.apply(
        decoded.samples,
        decoded.sample_rate,
        threshold=settings.vad_threshold,
        min_silence_ms=settings.vad_min_silence_ms,
        speech_pad_ms=settings.vad_speech_pad_ms,
        min_speech_sec=settings.min_speech_sec,
    )
    emb = embedding_svc.extract(
        vad_result.samples,
        model_name=settings.embedding_model,
        cache_dir=settings.model_cache_dir,
    )

    elapsed_ms = (time.perf_counter() - started) * 1000
    return ExtractResponse(
        audio=AudioInfo(
            duration_sec=round(vad_result.total_duration_sec, 3),
            speech_duration_sec=round(vad_result.speech_duration_sec, 3),
            speech_ratio=round(vad_result.speech_ratio, 3),
            sample_rate=decoded.sample_rate,
            source_sample_rate=decoded.source_sample_rate,
            source_channels=decoded.source_channels,
            segments=[
                SpeechSegmentOut(start=round(s.start, 3), end=round(s.end, 3))
                for s in vad_result.segments
            ],
        ),
        embedding=EmbeddingInfo(
            vector=emb.vector,
            dim=emb.dim,
            model=emb.model,
            l2_normalized=emb.l2_normalized,
        ),
        elapsed_ms=round(elapsed_ms, 1),
    )


@router.post(
    "/extract",
    response_model=ExtractResponse,
    tags=["voiceprint"],
    summary="오디오에서 화자 임베딩(성문) 추출",
    responses={422: {"description": "오디오 반려 — 응답 본문의 code로 사유 구분"}},
)
async def extract(
    file: UploadFile = File(..., description="16kHz 16-bit Mono WAV"),
    settings: Settings = Depends(get_settings),
) -> ExtractResponse:
    """업로드된 오디오에서 성문 벡터를 추출한다.

    파이프라인: 디코딩 → VAD(무음 제거) → ECAPA-TDNN 임베딩.

    유효 발화가 기준에 미달하면 422와 함께 사유 코드를 반환한다. 클라이언트는
    사유별로 다른 재녹음 안내를 띄운다 (04 §4).
    """
    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise AudioRejected(
            ErrorCode.FILE_TOO_LARGE,
            f"파일이 너무 큽니다. 최대 {settings.max_upload_bytes // (1024 * 1024)}MB까지 허용합니다.",
        )

    # 추론은 GIL을 오래 잡는 CPU 작업이다. 워커 스레드로 넘겨 이벤트 루프가
    # 다른 요청의 I/O를 계속 처리하게 한다.
    return await anyio.to_thread.run_sync(_run_pipeline, raw, settings)
