"""반려 사유 코드와 예외 정의.

클라이언트는 사유 코드별로 다른 재녹음 안내를 띄운다 (04 §4 "서버 반려 처리").
따라서 사유 코드는 UI 문구와 1:1로 대응하는 안정적인 식별자여야 하며,
값을 바꾸면 클라이언트가 깨진다.
"""

from enum import Enum


class ErrorCode(str, Enum):
    """오디오 반려 사유. 값은 클라이언트 계약이므로 변경 금지."""

    EMPTY_FILE = "empty_file"
    """업로드 본문이 비어 있음."""

    FILE_TOO_LARGE = "file_too_large"
    """허용 크기 초과."""

    UNREADABLE_AUDIO = "unreadable_audio"
    """디코딩 실패 — 손상되었거나 지원하지 않는 포맷."""

    AUDIO_TOO_LONG = "audio_too_long"
    """단일 요청 허용 길이 초과."""

    NO_SPEECH_DETECTED = "no_speech_detected"
    """VAD가 발화를 전혀 찾지 못함 — 무음이거나 소음뿐."""

    SPEECH_TOO_SHORT = "speech_too_short"
    """유효 발화가 최소 길이 미달 — 더 길게 말해야 함."""


class AudioRejected(Exception):
    """오디오가 임베딩 추출 조건을 만족하지 못해 반려됨.

    HTTP 422로 변환되며, 본문에 `code`와 `detail`을 담는다.
    서버 장애가 아니라 입력 문제이므로 5xx가 아니다.
    """

    def __init__(self, code: ErrorCode, detail: str, **context: object) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.context = context
