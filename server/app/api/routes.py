"""API 라우트.

범위: 성문 추출(`/extract`), 등록(`/enroll`), 1:1 검증(`/verify`).
검증은 AS-Norm 정규화 점수로 판정하며, 코호트가 없으면 원시 코사인으로 폴백한다.

경로에 버전을 박아두는 이유는 검증 API가 이후 기능 보강이 예정된 확장 지점이기
때문이다 (01 §2).
"""

from __future__ import annotations

import functools
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
    SeparationInfo,
    SpoofInfo,
    SpeechSegmentOut,
    VerifyResponse,
)
from app.services import antispoof as antispoof_svc
from app.services import audio as audio_svc
from app.services import embedding as embedding_svc
from app.services import enhance as enhance_svc
from app.services import scoring
from app.services import separation as separation_svc
from app.services import vad as vad_svc

logger = logging.getLogger(__name__)
router = APIRouter()

# 추론 동시 실행 상한. CPU 바운드 작업이라 무제한으로 풀면 처리량은 그대로인데
# 지연만 늘어난다 (config.max_concurrent_inference 주석의 실측 참조).
# 기동 시 설정값으로 초기화한다.
_inference_slots: anyio.Semaphore | None = None


def configure_inference_limit(limit: int) -> None:
    """추론 동시 실행 상한을 설정한다 (기동 시 1회)."""
    global _inference_slots
    _inference_slots = anyio.Semaphore(limit) if limit > 0 else None


async def _run_inference(func, *args, **kwargs):
    """추론을 워커 스레드에서 실행하되 동시 실행 수를 제한한다.

    세마포어를 스레드 진입 **전에** 잡는다. 스레드를 먼저 띄우고 안에서 기다리면
    스레드만 쌓여 메모리와 스케줄링 비용이 든다.
    """
    call = functools.partial(func, *args, **kwargs)
    if _inference_slots is None:
        return await anyio.to_thread.run_sync(call)
    async with _inference_slots:
        return await anyio.to_thread.run_sync(call)


@dataclass(frozen=True)
class _Analyzed:
    """전처리 + 임베딩 결과 묶음."""

    audio: AudioInfo
    embedding: embedding_svc.SpeakerEmbedding
    speech_duration_sec: float
    separation: SeparationInfo | None = None
    spoof: SpoofInfo | None = None


def _analyze(
    raw: bytes,
    settings: Settings,
    *,
    reference_embedding: list[float] | None = None,
) -> _Analyzed:
    """디코딩 → (분리) → (향상) → VAD → 임베딩.

    CPU 바운드이므로 워커 스레드에서 실행된다.

    `reference_embedding`이 주어지고 분리가 켜져 있으면, 혼합에서 그 화자에
    해당하는 출력을 골라 이후 단계에 넘긴다 (Phase 7). 등록 성문이 없으면
    분리해도 어느 출력이 타겟인지 알 수 없으므로 건너뛴다.
    """
    decoded = audio_svc.decode(
        raw,
        target_rate=settings.target_sample_rate,
        max_sec=settings.max_audio_sec,
    )
    samples = decoded.samples
    separation_info: SeparationInfo | None = None
    spoof_info: SpoofInfo | None = None

    # 딥페이크 탐지를 **가장 앞**에 둔다 (02 §5). 합성 음성이면 이후의 분리·
    # 임베딩·대조가 모두 무의미하므로, 비싼 연산을 하기 전에 걸러낸다.
    #
    # 분리보다 앞인 이유: 분리는 원본에 없던 아티팩트를 만들어 탐지기를 혼란시킬
    # 수 있다. 판정은 사용자가 실제로 보낸 신호를 대상으로 해야 한다.
    if settings.antispoof_enabled:
        spoof_result = antispoof_svc.detect(
            samples,
            weights_path=settings.antispoof_weights,
            threshold=settings.antispoof_threshold,
        )
        spoof_info = SpoofInfo(
            applied=True,
            spoof_score=round(spoof_result.spoof_score, 6),
            threshold=spoof_result.threshold,
            segments_scored=spoof_result.segments_scored,
        )
        if spoof_result.is_spoof:
            # 사용자에게는 점수를 알리지 않는다. 공격자가 점수를 보고 우회
            # 방법을 탐색하는 것을 막기 위해서다 (FR-18).
            raise AudioRejected(
                ErrorCode.SPOOF_DETECTED,
                "실제 음성으로 확인되지 않았습니다. 직접 말씀해주세요.",
                spoof_score=spoof_result.spoof_score,
            )

    if settings.separation_enabled and reference_embedding is not None:
        # 분리를 가장 앞에 둔다. 겹친 화자를 먼저 떼어내야 이후의 향상·VAD·임베딩이
        # 단일 화자를 대상으로 동작한다.
        def _embed_source(source):
            return embedding_svc.extract(
                source,
                model_name=settings.embedding_model,
                cache_dir=settings.model_cache_dir,
                backend=settings.embedding_backend,
                onnx_threads=settings.onnx_intra_op_threads,
            ).vector

        result = separation_svc.extract_target(
            samples,
            reference_embedding,
            embed_fn=_embed_source,
            model_name=settings.separation_model,
            cache_dir=settings.model_cache_dir,
        )
        samples = result.target
        margin = result.selection_margin
        if (
            settings.separation_min_margin > 0
            and margin is not None
            and margin < settings.separation_min_margin
        ):
            logger.warning(
                "타겟 선택이 모호합니다 (마진 %.4f < %.4f). 판정 신뢰도가 낮습니다.",
                margin,
                settings.separation_min_margin,
            )
        separation_info = SeparationInfo(
            applied=True,
            source_count=result.source_count,
            target_index=result.target_index,
            target_similarity=round(result.target_similarity, 6),
            selection_margin=round(margin, 6) if margin is not None else None,
        )

    if settings.enhance_enabled:
        # VAD 앞에 둔다. 소음을 먼저 걷어내야 VAD가 발화 구간을 더 정확히 잡는다.
        samples = enhance_svc.apply(samples, decoded.sample_rate)

    vad_result = vad_svc.apply(
        samples,
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
        backend=settings.embedding_backend,
        onnx_threads=settings.onnx_intra_op_threads,
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
        separation=separation_info,
        spoof=spoof_info,
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
        models_loaded=embedding_svc.is_loaded(),
        storage=db_session.backend_name(),
        storage_ok=await repo.health(),
        asnorm_active=cohort is not None,
        cohort_size=cohort.size if cohort else 0,
        separation_active=settings.separation_enabled and separation_svc.is_loaded(),
        antispoof_active=settings.antispoof_enabled and antispoof_svc.is_loaded(),
        enhance_active=settings.enhance_enabled and enhance_svc.is_loaded(),
        embedding_backend=settings.embedding_backend,
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
    analyzed = await _run_inference(_analyze, raw, settings)

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
    analyzed = await _run_inference(_analyze, raw, settings)

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

    # 등록 성문을 먼저 읽는다. 분리를 켠 경우 혼합에서 어느 출력이 타겟인지
    # 고르려면 등록 임베딩이 필요하기 때문이다 (Phase 7). 오디오 분석 전에
    # 미등록·모델 불일치를 걸러내면 헛된 추론 비용도 아낀다.
    enrollments = await repo.list_active_enrollments(user_id)
    if not enrollments:
        await _log("not_enrolled", error_code=ErrorCode.NOT_ENROLLED.value)
        raise AudioRejected(
            ErrorCode.NOT_ENROLLED,
            f"등록된 성문이 없습니다. 먼저 등록해주세요. (user_id={user_id})",
            user_id=user_id,
        )

    # 모델이 다른 벡터끼리 비교하면 의미 없는 유사도가 나온다. 조용히 통과시키면
    # 그 값이 그대로 인증 판정에 쓰이므로 명시적으로 거부한다 (02 §6).
    current_model = settings.embedding_model
    usable = [
        e
        for e in enrollments
        if e.model == current_model and e.dim == settings.embedding_dim
    ]
    if not usable:
        stale = sorted({e.model for e in enrollments})
        await _log(
            "model_mismatch",
            error_code=ErrorCode.MODEL_MISMATCH.value,
            model=current_model,
        )
        raise AudioRejected(
            ErrorCode.MODEL_MISMATCH,
            (
                f"등록 성문이 현재 모델과 호환되지 않습니다. 재등록이 필요합니다. "
                f"(등록={stale}, 현재={current_model})"
            ),
            user_id=user_id,
        )

    # 분리 시 타겟 선택 기준으로 쓸 등록 성문. 여러 개면 가장 최근 것을 쓴다
    # (list_active_enrollments가 최신순으로 반환한다).
    reference = usable[0].embedding if settings.separation_enabled else None

    try:
        analyzed = await _run_inference(
            _analyze, raw, settings, reference_embedding=reference
        )
    except AudioRejected as exc:
        outcome = (
            "spoof_detected"
            if exc.code is ErrorCode.SPOOF_DETECTED
            else "audio_rejected"
        )
        await _log(
            outcome,
            error_code=exc.code.value,
            spoof_score=exc.context.get("spoof_score"),
        )
        raise

    probe = analyzed.embedding

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
        separation=analyzed.separation,
        spoof=analyzed.spoof,
        audio=analyzed.audio,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )
