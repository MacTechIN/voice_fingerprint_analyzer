"""API 요청·응답 스키마.

응답은 확장 가능 스키마로 설계한다 (01 §2, 06 Phase 2 설계 원칙).
Phase B~D에서 `normalized_score`, `spoof_score` 등이 추가되므로 클라이언트는
알 수 없는 필드를 무시하도록 구현해야 한다 (Tolerant Reader).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SpeechSegmentOut(BaseModel):
    """VAD가 검출한 발화 구간 (초)."""

    start: float = Field(..., description="구간 시작 시각(초)")
    end: float = Field(..., description="구간 종료 시각(초)")


class AudioInfo(BaseModel):
    """입력 오디오와 전처리 결과 요약.

    클라이언트가 녹음 품질을 스스로 진단하고 사용자에게 안내할 수 있도록
    노출한다 (04 §4).
    """

    duration_sec: float = Field(..., description="입력 오디오 전체 길이")
    speech_duration_sec: float = Field(..., description="VAD 통과 유효 발화 길이")
    speech_ratio: float = Field(..., description="전체 대비 발화 비율 (0~1)")
    sample_rate: int = Field(..., description="처리 샘플레이트")
    source_sample_rate: int = Field(..., description="업로드된 원본 샘플레이트")
    source_channels: int = Field(..., description="업로드된 원본 채널 수")
    segments: list[SpeechSegmentOut] = Field(
        default_factory=list, description="검출된 발화 구간 목록"
    )


class EmbeddingInfo(BaseModel):
    """임베딩과 그 출처.

    `model`/`dim`은 벡터와 반드시 함께 저장한다. 모델 교체 시 기존 벡터의
    재등록 대상을 식별하는 유일한 근거다 (02 §6).
    """

    vector: list[float] = Field(..., description="화자 임베딩 벡터")
    dim: int = Field(..., description="임베딩 차원")
    model: str = Field(..., description="임베딩 모델 식별자")
    l2_normalized: bool = Field(..., description="L2 정규화 적용 여부")


class ExtractResponse(BaseModel):
    """`POST /api/v1/extract` 성공 응답."""

    status: str = Field(default="success")
    audio: AudioInfo
    embedding: EmbeddingInfo
    elapsed_ms: float = Field(..., description="서버 처리 소요 시간(ms)")


class ErrorResponse(BaseModel):
    """반려·오류 응답.

    `code`는 클라이언트가 사유별 재녹음 안내를 분기하는 안정적 식별자다.
    """

    status: str = Field(default="error")
    code: str = Field(..., description="사유 코드 (app.core.errors.ErrorCode)")
    detail: str = Field(..., description="사람이 읽는 설명")


class EnrollResponse(BaseModel):
    """`POST /api/v1/enroll` 성공 응답."""

    status: str = Field(default="success")
    user_id: str
    enrollment_id: int = Field(..., description="저장된 등록 성문 ID")
    replaced: int = Field(0, description="이번 등록으로 비활성화된 기존 성문 수")
    audio: AudioInfo
    embedding: EmbeddingInfo
    elapsed_ms: float


class VerifyResponse(BaseModel):
    """`POST /api/v1/verify` 성공 응답.

    Phase B~D에서 `normalized_score`(AS-Norm), `spoof_score`(딥페이크 탐지)가
    최상위에 추가된다. 클라이언트는 알 수 없는 필드를 무시해야 한다.
    """

    status: str = Field(default="success")
    user_id: str
    is_verified: bool = Field(..., description="임계값 초과 여부 (FR-06)")
    match_probability: float = Field(
        ...,
        description=(
            "0~100 척도의 일치도. 판정 경계를 50%에 맞춘 구간 선형 재척도이며 "
            "확률이 아니다 — 캘리브레이션은 Phase 6 과제."
        ),
    )
    raw_cosine: float = Field(..., description="원시 코사인 유사도")
    threshold: float = Field(..., description="판정에 사용된 임계값")
    compared_enrollments: int = Field(..., description="대조한 등록 성문 수")
    audio: AudioInfo
    elapsed_ms: float


class HealthResponse(BaseModel):
    """헬스 체크 응답."""

    status: str
    model: str
    models_loaded: bool
    storage: str = Field("unknown", description="저장소 백엔드 (postgres | memory)")
    storage_ok: bool = Field(False, description="저장소 접속 가능 여부")
