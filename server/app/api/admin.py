"""관리자 대시보드 API (03 웹앱 정의서).

**인증이 필요하다.** 이 엔드포인트들은 사용자 ID·클라이언트 IP·인증 이력을
노출한다. `VG_ADMIN_TOKEN`이 설정되지 않으면 **503으로 거부**한다 — 토큰을
안 걸었을 때 조용히 열어두는 것보다 아예 막히는 편이 안전하다.
"""

from __future__ import annotations

import json
import logging
import secrets
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.config import Settings, get_settings
from app.db import session as db_session
from app.db.analytics import Analytics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """관리자 토큰 검사.

    토큰 미설정 시 열어두지 않고 막는 이유: 감사 로그와 사용자 ID가 노출되는
    엔드포인트이므로, 설정을 빠뜨린 배포가 조용히 공개되면 안 된다.
    """
    expected = settings.admin_token
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=(
                "관리자 API가 비활성화되어 있습니다. "
                "VG_ADMIN_TOKEN을 설정한 뒤 다시 시도하세요."
            ),
        )

    header = request.headers.get("authorization", "")
    provided = header[7:] if header.lower().startswith("bearer ") else ""
    # 타이밍 공격을 피하려 상수 시간 비교를 쓴다
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="관리자 인증에 실패했습니다.")


def get_analytics() -> Analytics:
    """집계 질의 핸들. Postgres가 아니면 사용할 수 없다."""
    pool = db_session.get_pool()
    if pool is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "대시보드는 PostgreSQL 저장소에서만 동작합니다. "
                "현재 인메모리 저장소로 기동되어 통계를 제공할 수 없습니다."
            ),
        )
    return Analytics(pool)


@router.get("/overview", dependencies=[Depends(require_admin)])
async def overview(
    hours: Optional[int] = Query(None, ge=1, le=8760, description="조회 기간(시간). 미지정 시 전체"),
    settings: Settings = Depends(get_settings),
    analytics: Analytics = Depends(get_analytics),
) -> dict:
    """대시보드 상단 요약 (03 §3)."""
    data = await analytics.overview(hours=hours, current_model=settings.embedding_model)
    return {
        "total_attempts": data.total_attempts,
        "outcomes": [{"outcome": o.outcome, "count": o.count} for o in data.outcomes],
        "verified_count": data.verified_count,
        "rejected_count": data.rejected_count,
        "audio_rejected_count": data.audio_rejected_count,
        "decision_count": data.decision_count,
        "verified_ratio": data.verified_ratio,
        "latency": {
            "avg_ms": data.latency_avg_ms,
            "p50_ms": data.latency_p50_ms,
            "p95_ms": data.latency_p95_ms,
        },
        "enrollments": {
            "total": data.total_enrollments,
            "active": data.active_enrollments,
            "users": data.enrolled_users,
        },
        "cohort_size": data.cohort_size,
        "first_attempt_at": _iso(data.first_attempt_at),
        "last_attempt_at": _iso(data.last_attempt_at),
        # 지표를 오해하지 않도록 서버가 직접 단서를 붙인다
        "caveat": (
            "verified_ratio는 정확도가 아니다. 감사 로그에는 정답 레이블이 없으므로 "
            "시도자가 실제 본인이었는지 알 수 없다. FAR/FRR은 레이블이 있는 "
            "오프라인 평가(eval/)에서만 산출된다."
        ),
    }


@router.get("/timeseries", dependencies=[Depends(require_admin)])
async def timeseries(
    hours: int = Query(24, ge=1, le=8760),
    bucket_minutes: int = Query(60, ge=1, le=1440),
    analytics: Analytics = Depends(get_analytics),
) -> dict:
    """시간대별 요청 추이와 응답 시간 (03 §3)."""
    buckets = await analytics.timeseries(hours=hours, bucket_minutes=bucket_minutes)
    return {
        "hours": hours,
        "bucket_minutes": bucket_minutes,
        "buckets": [
            {
                "bucket": _iso(b.bucket),
                "verified": b.verified,
                "rejected": b.rejected,
                "audio_rejected": b.audio_rejected,
                "not_enrolled": b.not_enrolled,
                "other": b.other,
                "total": b.total,
                "avg_elapsed_ms": b.avg_elapsed_ms,
            }
            for b in buckets
        ],
    }


@router.get("/score-distribution", dependencies=[Depends(require_admin)])
async def score_distribution(
    normalized: bool = Query(True, description="AS-Norm 점수 기준 여부"),
    bins: int = Query(20, ge=4, le=100),
    analytics: Analytics = Depends(get_analytics),
) -> dict:
    """판정 점수 히스토그램 (03 §4).

    genuine/impostor가 아니라 **판정 결과**로 나눈 분포다. 운영 로그에는 정답
    레이블이 없기 때문이다.
    """
    dist = await analytics.score_distribution(use_normalized=normalized, bins=bins)
    return {
        "field": dist.field_name,
        "threshold": dist.threshold,
        "sample_count": dist.sample_count,
        "min_score": dist.min_score,
        "max_score": dist.max_score,
        "bins": [
            {
                "lower": b.lower,
                "upper": b.upper,
                "verified": b.verified,
                "rejected": b.rejected,
                "total": b.total,
            }
            for b in dist.bins
        ],
        "caveat": (
            "판정 결과별 분포이며 genuine/impostor 분포가 아니다. "
            "운영 로그에는 정답 레이블이 없다."
        ),
    }


@router.get("/threshold-impact", dependencies=[Depends(require_admin)])
async def threshold_impact(
    thresholds: str = Query(..., description="쉼표로 구분한 후보 임계값 (예: 2.0,2.9673,4.0)"),
    normalized: bool = Query(True),
    analytics: Analytics = Depends(get_analytics),
) -> dict:
    """임계값 변경 시 과거 판정이 어떻게 뒤집히는지 (FR-11 근거).

    **오류율이 아니라 영향도다.** 통과가 거부로 바뀐다는 것은 셀 수 있어도,
    그 변경이 옳은지는 레이블 없이 말할 수 없다.
    """
    try:
        candidates = [float(v) for v in thresholds.split(",") if v.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="임계값은 숫자여야 합니다.")
    if not candidates or len(candidates) > 50:
        raise HTTPException(status_code=400, detail="후보 임계값은 1~50개여야 합니다.")

    impacts = await analytics.threshold_impact(candidates, use_normalized=normalized)
    return {
        "field": "normalized_score" if normalized else "raw_cosine",
        "impacts": [
            {
                "threshold": i.threshold,
                "would_pass": i.would_pass,
                "would_fail": i.would_fail,
                "flipped_from_pass": i.flipped_from_pass,
                "flipped_from_fail": i.flipped_from_fail,
                "total": i.total,
            }
            for i in impacts
        ],
        "caveat": (
            "판정이 뒤집히는 건수이며 오류율이 아니다. 임계값 결정은 레이블이 있는 "
            "평가(eval/calibrate.py)의 EER·minDCF를 근거로 해야 한다."
        ),
    }


@router.get("/speakers", dependencies=[Depends(require_admin)])
async def speakers(
    limit: int = Query(200, ge=1, le=1000),
    settings: Settings = Depends(get_settings),
    analytics: Analytics = Depends(get_analytics),
) -> dict:
    """화자 데이터베이스 (03 §3).

    원본 오디오는 분석 후 즉시 파기되므로 여기에 없고, 벡터도 반환하지 않는다.
    관리자가 알아야 할 것은 **어떤 모델로 언제 등록됐고 재등록이 필요한가**이다.
    """
    rows = await analytics.speakers(current_model=settings.embedding_model, limit=limit)
    return {
        "current_model": settings.embedding_model,
        "current_dim": settings.embedding_dim,
        "needs_reenrollment_count": sum(1 for r in rows if r.needs_reenrollment),
        "speakers": [
            {
                "user_id": r.user_id,
                "active_count": r.active_count,
                "inactive_count": r.inactive_count,
                "model": r.model,
                "dim": r.dim,
                "last_enrolled_at": _iso(r.last_enrolled_at),
                "avg_speech_duration_sec": r.avg_speech_duration_sec,
                "needs_reenrollment": r.needs_reenrollment,
            }
            for r in rows
        ],
    }


@router.get("/attempts", dependencies=[Depends(require_admin)])
async def attempts(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user_id: Optional[str] = None,
    outcome: Optional[str] = None,
    analytics: Analytics = Depends(get_analytics),
) -> dict:
    """오딧 트레일 (03 §3).

    원시 점수와 정규화 점수를 함께 반환해 정규화 효과를 추적할 수 있게 한다.
    """
    page = await analytics.attempts(
        limit=limit, offset=offset, user_id=user_id, outcome=outcome
    )
    return {
        "total": page.total,
        "limit": page.limit,
        "offset": page.offset,
        "has_more": page.has_more,
        "attempts": [
            {
                "id": r.id,
                "user_id": r.user_id,
                "outcome": r.outcome,
                "is_verified": r.is_verified,
                "raw_cosine": r.raw_cosine,
                "normalized_score": r.normalized_score,
                "match_probability": r.match_probability,
                "threshold": r.threshold,
                "model": r.model,
                "speech_duration_sec": r.speech_duration_sec,
                "error_code": r.error_code,
                "client_ip": r.client_ip,
                "elapsed_ms": r.elapsed_ms,
                "created_at": _iso(r.created_at),
            }
            for r in page.rows
        ],
    }


@router.get("/calibration", dependencies=[Depends(require_admin)])
async def calibration(settings: Settings = Depends(get_settings)) -> dict:
    """현재 판정 설정과 오프라인 캘리브레이션 결과 (FR-11, FR-12).

    EER·minDCF는 **레이블이 있는 오프라인 평가에서만** 나온다. 운영 로그로는
    계산할 수 없으므로, `eval/calibrate.py`가 남긴 보고서를 그대로 읽어 보여준다.
    보고서가 없으면 캘리브레이션을 아직 돌리지 않았다는 뜻이다.
    """
    cohort = db_session.get_cohort()
    report = _load_calibration_report(settings)

    return {
        "active": {
            "embedding_backend": settings.embedding_backend,
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "asnorm_enabled": settings.asnorm_enabled,
            "asnorm_active": cohort is not None,
            "asnorm_top_k": settings.asnorm_top_k,
            "asnorm_threshold": settings.asnorm_threshold,
            "match_threshold": settings.match_threshold,
            "cohort_size": cohort.size if cohort else 0,
        },
        "report": report,
        # 임계값은 환경변수로 주입된다. 런타임 변경 API를 두지 않은 이유는
        # 인증 파라미터를 원격에서 바꾸는 것이 그 자체로 공격면이기 때문이다.
        "how_to_apply": {
            "asnorm_threshold": "VG_ASNORM_THRESHOLD",
            "match_threshold": "VG_MATCH_THRESHOLD",
            "asnorm_top_k": "VG_ASNORM_TOP_K",
            "note": "환경변수를 바꾼 뒤 서버를 재시작한다. 런타임 변경 API는 제공하지 않는다.",
        },
    }


def _load_calibration_report(settings: Settings) -> Optional[dict]:
    """`eval/calibrate.py`가 남긴 보고서를 현재 백엔드 기준으로 찾는다."""
    slug = f"{settings.embedding_backend}__{settings.embedding_model.replace('/', '_')}"
    path = Path(__file__).resolve().parent.parent.parent / ".data" / f"calibration_report__{slug}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        logger.exception("캘리브레이션 보고서를 읽지 못했습니다: %s", path)
        return None


def _iso(value) -> Optional[str]:
    return value.isoformat() if value is not None else None
