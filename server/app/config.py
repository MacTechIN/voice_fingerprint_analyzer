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

    # --- 저장소 (Phase 2) ---
    database_url: str = ""
    """PostgreSQL 접속 URL. 비우면 인메모리 저장소로 뜬다(개발·데모용).

    예: postgresql://voiceguard:voiceguard@127.0.0.1:54321/voiceguard
    """

    # --- 판정 (Phase 2) ---
    match_threshold: float = 0.3333
    """원시 코사인 판정 임계값 (AS-Norm 미적용 시 폴백 경로에서 사용).

    LibriSpeech dev-clean 40화자 2,400트라이얼의 EER 지점 (EER 1.67%).
    이전 기본값 0.25는 관례값이었고, 같은 트라이얼에서 **FAR 6.67%**를 냈다 —
    인증 시스템에서 100번 중 6~7번 타인을 통과시키는 수준이다.
    """

    # --- AS-Norm (Phase 6) ---
    asnorm_enabled: bool = True
    """AS-Norm 점수 정규화 사용 여부.

    코호트가 비어 있으면 자동으로 비활성화되고 원시 코사인으로 판정한다.
    그 사실은 `/health`의 `asnorm_active`에 드러난다 — 정규화가 꺼진 채로
    운영되는 것을 모르고 지나치면 안 된다.
    """

    asnorm_top_k: int = 200
    """코호트 상위 K개를 적응적으로 선택한다.

    클수록 통계가 안정되지만 '가장 혼동하기 쉬운 임포스터'라는 적응적 성격이
    옅어진다. 캘리브레이션에서 K=100/200/300의 EER이 1.25%로 동률이었고,
    그중 minDCF가 가장 낮은(0.0317) K=200을 택했다. minDCF는 사칭 시도가 드문
    실제 환경을 반영해 FAR에 더 큰 가중을 두므로 인증 시스템에 더 적합한 기준이다.
    """

    asnorm_threshold: float = 3.0033
    """AS-Norm 정규화 점수의 판정 임계값 (K=200의 EER 지점, EER 1.25%).

    정규화 점수는 코호트 기준 표준화 값이라 원시 코사인과 척도가 완전히 다르다
    (코사인처럼 [-1,1]에 갇히지 않는다). 두 임계값을 서로 바꿔 쓰면 안 된다.

    더 보수적인 운영(오수락 억제 우선)을 원하면 minDCF 지점인 4.2009를 쓴다.
    """

    enroll_replaces_existing: bool = True
    """등록 시 기존 성문을 비활성화할지 여부.

    기본값 True는 "마지막 등록이 유효한 성문"이라는 단순한 모델이다. False로
    두면 여러 발화가 누적되고 검증은 그중 최대 유사도로 판정한다.
    """

    # --- 런타임 ---
    warmup_on_startup: bool = True
    """기동 시 모델을 미리 적재해 첫 요청 지연을 없앤다."""


@lru_cache
def get_settings() -> Settings:
    return Settings()
