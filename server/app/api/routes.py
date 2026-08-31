"""API 라우트.

범위: 성문 추출(`/extract`), 등록(`/enroll`), 1:1 검증(`/verify`).
검증은 AS-Norm 정규화 점수로 판정하며, 코호트가 없으면 원시 코사인으로 폴백한다.

경로에 버전을 박아두는 이유는 검증 API가 이후 기능 보강이 예정된 확장 지점이기
때문이다 (01 §2).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import anyio
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.config import Settings, get_settings
from app.core.errors import AudioRejected, ErrorCode
from app.db import session as db_session
from app.db.repository import SpeakerRepository, VerificationLog
from app.schemas import (
    AudioInfo,
    EmbeddingInfo,
    EnrollResponse,
    ExtractResponse,
    HealthResponse,
    SpeechSegmentOut,
    VerifyResponse,
)
from app.services import audio as audio_svc
from app.services import embedding as embedding_svc
from app.services import scoring
from app.services import vad as vad_svc

logger = logging.getLogger(__name__)
router = APIRouter()


@dataclass(frozen=True)
class _Analyzed:
    """전처리 + 임베딩 결과 묶음."""

    audio: AudioInfo
    embedding: embedding_svc.SpeakerEmbedding
    speech_duration_sec: float


def _analyze(raw: bytes, settings: Settings) -> _Analyzed:
    """디코딩 → VAD → 임베딩. CPU 바운드이므로 워커 스레드에서 실행된다."""
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
    return _Analyzed(
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
        embedding=emb,
        speech_duration_sec=vad_result.speech_duration_sec,
    )


async def _read_upload(file: UploadFile, settings: Settings) -> bytes:
    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise AudioRejected(
            ErrorCode.FILE_TOO_LARGE,
            f"파일이 너무 큽니다. 최대 {settings.max_upload_bytes // (1024 * 1024)}MB까지 허용합니다.",
        )
    return raw


def _to_embedding_info(emb: embedding_svc.SpeakerEmbedding) -> EmbeddingInfo:
    return EmbeddingInfo(
        vector=emb.vector, dim=emb.dim, model=emb.model, l2_normalized=emb.l2_normalized
    )


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(
    settings: Settings = Depends(get_settings),
    repo: SpeakerRepository = Depends(db_session.get_repository),
) -> HealthResponse:
    """서비스 생존, 모델 적재, 저장소 접속 상태."""
    cohort = db_session.get_cohort()
    return HealthResponse(
        status="ok",
        model=settings.embedding_model,
        models_loaded=embedding_svc._classifier is not None,
        storage=db_session.backend_name(),
        storage_ok=await repo.health(),
        asnorm_active=cohort is not None,
        cohort_size=cohort.size if cohort else 0,
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
    """업로드된 오디오에서 성문 벡터를 추출한다 (저장하지 않는다).

    파이프라인: 디코딩 → VAD(무음 제거) → ECAPA-TDNN 임베딩.
    """
    started = time.perf_counter()
    raw = await _read_upload(file, settings)
    analyzed = await anyio.to_thread.run_sync(_analyze, raw, settings)

    return ExtractResponse(
        audio=analyzed.audio,
        embedding=_to_embedding_info(analyzed.embedding),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )


@router.post(
    "/enroll",
    response_model=EnrollResponse,
    tags=["voiceprint"],
    summary="사용자 성문 등록",
    responses={422: {"description": "오디오 반려 — 응답 본문의 code로 사유 구분"}},
)
async def enroll(
    user_id: str = Form(..., description="등록 대상 사용자 ID"),
    file: UploadFile = File(..., description="16kHz 16-bit Mono WAV"),
    settings: Settings = Depends(get_settings),
    repo: SpeakerRepository = Depends(db_session.get_repository),
) -> EnrollResponse:
    """오디오에서 성문을 추출해 사용자에 등록한다 (FR-04).

    `VG_ENROLL_REPLACES_EXISTING`(기본 true)이면 기존 등록을 비활성화한다.
    비활성화는 삭제가 아니므로 이전 벡터의 이력은 남는다.
    """
    started = time.perf_counter()
    raw = await _read_upload(file, settings)
    analyzed = await anyio.to_thread.run_sync(_analyze, raw, settings)

    replaced = 0
    if settings.enroll_replaces_existing:
        replaced = await repo.deactivate_enrollments(user_id)

    saved = await repo.add_enrollment(
        user_id=user_id,
        embedding=analyzed.embedding.vector,
        model=analyzed.embedding.model,
        dim=analyzed.embedding.dim,
        l2_normalized=analyzed.embedding.l2_normalized,
        speech_duration_sec=analyzed.speech_duration_sec,
    )
    logger.info(
        "성문 등록: user_id=%s enrollment_id=%s replaced=%s", user_id, saved.id, replaced
    )

    return EnrollResponse(
        user_id=user_id,
        enrollment_id=saved.id,
        replaced=replaced,
        audio=analyzed.audio,
        embedding=_to_embedding_info(analyzed.embedding),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )


@router.post(
    "/verify",
    response_model=VerifyResponse,
    tags=["voiceprint"],
    summary="사용자 성문 1:1 검증",
    responses={422: {"description": "오디오 반려 또는 미등록 — code로 사유 구분"}},
)
async def verify(
    request: Request,
    user_id: str = Form(..., description="검증 대상 사용자 ID"),
    file: UploadFile = File(..., description="16kHz 16-bit Mono WAV"),
    settings: Settings = Depends(get_settings),
    repo: SpeakerRepository = Depends(db_session.get_repository),
) -> VerifyResponse:
    """업로드된 음성이 해당 사용자와 동일인인지 판정한다 (FR-05, FR-06, FR-09).

    임포스터 코호트가 적재되어 있으면 AS-Norm 정규화 점수로 판정한다. 원시
    코사인에 고정 임계값을 적용하는 방식은 화자 상태·발화 길이·잔존 소음에 따라
    점수 편차가 커서 실무 실패율이 높다 (02 §4.2).

    코호트가 없으면 원시 코사인으로 폴백하되, `scoring_method` 필드와 `/health`의
    `asnorm_active`에 그 사실을 드러낸다.
    """
    started = time.perf_counter()
    client_ip = request.client.host if request.client else None

    async def _log(outcome: str, **fields) -> None:
        """감사 로그 기록. 실패해도 검증 응답을 막지 않는다 (03 오딧 트레일)."""
        try:
            await repo.log_verification(
                VerificationLog(
                    user_id=user_id,
                    outcome=outcome,
                    client_ip=client_ip,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
                    **fields,
                )
            )
        except Exception:
            logger.exception("검증 감사 로그 기록 실패 — 응답은 계속 진행한다")

    raw = await _read_upload(file, settings)

    try:
        analyzed = await anyio.to_thread.run_sync(_analyze, raw, settings)
    except AudioRejected as exc:
        await _log("audio_rejected", error_code=exc.code.value)
        raise

    enrollments = await repo.list_active_enrollments(user_id)
    if not enrollments:
        await _log(
            "not_enrolled",
            error_code=ErrorCode.NOT_ENROLLED.value,
            model=analyzed.embedding.model,
            speech_duration_sec=analyzed.speech_duration_sec,
        )
        raise AudioRejected(
            ErrorCode.NOT_ENROLLED,
            f"등록된 성문이 없습니다. 먼저 등록해주세요. (user_id={user_id})",
            user_id=user_id,
        )

    # 모델이 다른 벡터끼리 비교하면 의미 없는 유사도가 나온다. 조용히 통과시키면
    # 그 값이 그대로 인증 판정에 쓰이므로 명시적으로 거부한다 (02 §6).
    probe = analyzed.embedding
    usable = [e for e in enrollments if e.model == probe.model and e.dim == probe.dim]
    if not usable:
        stale = sorted({e.model for e in enrollments})
        await _log(
            "model_mismatch",
            error_code=ErrorCode.MODEL_MISMATCH.value,
            model=probe.model,
            speech_duration_sec=analyzed.speech_duration_sec,
        )
        raise AudioRejected(
            ErrorCode.MODEL_MISMATCH,
            (
                f"등록 성문이 현재 모델과 호환되지 않습니다. 재등록이 필요합니다. "
                f"(등록={stale}, 현재={probe.model})"
            ),
            user_id=user_id,
        )

    # AS-Norm이 가능하면 정규화 점수로 판정하고, 코호트가 없으면 원시 코사인으로
    # 폴백한다. 폴백 사실은 응답의 scoring_method와 /health에 드러나므로 정규화가
    # 꺼진 채 운영되는 것을 모르고 지나치지 않는다.
    cohort = db_session.get_cohort() if settings.asnorm_enabled else None
    references = [e.embedding for e in usable]
    if cohort is not None:
        result = scoring.match_best_normalized(
            probe.vector, references, cohort=cohort, threshold=settings.asnorm_threshold
        )
    else:
        result = scoring.match_best(probe.vector, references, settings.match_threshold)

    await _log(
        "verified" if result.is_verified else "rejected",
        raw_cosine=result.raw_cosine,
        normalized_score=result.normalized_score,
        match_probability=result.match_probability,
        is_verified=result.is_verified,
        threshold=result.threshold,
        model=probe.model,
        speech_duration_sec=analyzed.speech_duration_sec,
    )
    logger.info(
        "성문 검증: user_id=%s verified=%s method=%s cosine=%.4f normalized=%s",
        user_id,
        result.is_verified,
        result.scoring_method,
        result.raw_cosine,
        f"{result.normalized_score:.4f}" if result.normalized_score is not None else "-",
    )

    return VerifyResponse(
        user_id=user_id,
        is_verified=result.is_verified,
        match_probability=round(result.match_probability, 1),
        raw_cosine=round(result.raw_cosine, 6),
        normalized_score=(
            round(result.normalized_score, 6) if result.normalized_score is not None else None
        ),
        scoring_method=result.scoring_method,
        threshold=result.threshold,
        compared_enrollments=len(usable),
        audio=analyzed.audio,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )
