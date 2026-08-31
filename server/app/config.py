"""서버 설정.

환경변수 또는 .env 파일로 오버라이드한다. 접두사는 `VG_`.
예: VG_MIN_SPEECH_SEC=2.0
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VG_", env_file=".env", extra="ignore")

    # --- 오디오 규격 (04_NativeApp_Definition.md와 일치시킬 것) ---
    target_sample_rate: int = 16_000
    """서버 내부 처리 샘플레이트. 입력이 다르면 리샘플링한다."""

    max_upload_bytes: int = 32 * 1024 * 1024
    """업로드 허용 최대 바이트 (32MB)."""

    max_audio_sec: float = 300.0
    """단일 요청 허용 최대 오디오 길이. 초과 시 반려 (긴 오디오는 Phase 7 청크 처리 대상)."""

    # --- VAD ---
    vad_threshold: float = 0.5
    """Silero VAD 발화 확률 임계값."""

    vad_min_silence_ms: int = 300
    """이보다 짧은 무음은 발화 구간을 끊지 않는다."""

    vad_speech_pad_ms: int = 30
    """검출된 발화 구간 앞뒤로 덧붙이는 여유. 어절 첫/끝 음소 잘림 방지."""

    min_speech_sec: float = 1.5
    """VAD 통과 유효 발화 길이 하한. 미달 시 재녹음 요청 (02 §2.2)."""

    # --- 임베딩 ---
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    """화자 임베딩 백본. Phase 6에서 WeSpeaker로 전환 예정 (02 §3.1)."""

    model_cache_dir: str = "./.model_cache"
    """사전학습 모델 다운로드 캐시 경로."""

    # --- 런타임 ---
    warmup_on_startup: bool = True
    """기동 시 모델을 미리 적재해 첫 요청 지연을 없앤다."""


@lru_cache
def get_settings() -> Settings:
    return Settings()
