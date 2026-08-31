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

    # --- 음성 향상 (Phase 6) ---
    enhance_enabled: bool = False
    """DeepFilterNet 소음 억제 적용 여부.

    **기본값이 비활성인 이유:** 향상은 공짜가 아니다. 이미 깨끗한 오디오에
    적용하면 아티팩트를 더해 임베딩 품질을 오히려 떨어뜨릴 수 있고, 48kHz
    업/다운샘플링 비용도 든다. 채택 여부는 `eval/noise_eval.py`의 실측으로
    판단한다.

    소음 환경이 확인된 배포에서만 켠다.
    """

    # --- 임베딩 ---
    embedding_backend: str = "wespeaker"
    """임베딩 백엔드: `speechbrain`(192차원) | `wespeaker`(ONNX, 256차원).

    백엔드를 바꾸면 임베딩 차원과 잠재 공간이 모두 달라져 기존 벡터와 호환되지
    않는다. 반드시 재등록과 코호트 재구축, 임계값 재캘리브레이션이 뒤따라야 한다.
    """

    embedding_model: str = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
    """화자 임베딩 백본 식별자.

    WeSpeaker로 전환한 근거는 정확도가 아니라 **속도와 배포 무게**다. 실측에서
    AS-Norm 적용 시 EER이 1.25%로 SpeechBrain과 동률이었고(2,400트라이얼에서
    0.17%p 차이는 오류 4건 수준의 통계적 잡음), 스레드 수를 맞춘 추론은
    1.8배 빨랐으며(58ms vs 104ms @ 4스레드) 모델 적재는 8배 빨랐다(0.25s vs 2.06s).
    ONNX 런타임이라 추론에 PyTorch가 필요 없다는 점도 배포에 유리하다.

    SpeechBrain으로 되돌리려면 `VG_EMBEDDING_BACKEND=speechbrain`,
    `VG_EMBEDDING_MODEL=speechbrain/spkrec-ecapa-voxceleb`, `VG_EMBEDDING_DIM=192`,
    그리고 아래 임계값을 SpeechBrain 캘리브레이션 값으로 되돌린다
    (원시 0.3333 / AS-Norm 3.0033).
    """

    onnx_intra_op_threads: int = 4
    """ONNX 백엔드의 연산 내 병렬 스레드 수.

    지연 시간과 동시 처리량의 트레이드오프다. 실측(5.9초 오디오, ResNet34-LM):
    1스레드 206ms / 2스레드 108ms / 4스레드 58ms / 8스레드 32ms.
    동시 요청이 많은 배포에서는 줄여 코어를 요청 간에 나눠 쓰는 편이 낫다.
    """

    embedding_dim: int = 256
    """임베딩 차원. DB 스키마(vector(N))와 반드시 일치해야 한다.

    speechbrain ECAPA-TDNN: 192 / WeSpeaker ResNet34-LM: 256
    """

    model_cache_dir: str = "./.model_cache"
    """사전학습 모델 다운로드 캐시 경로."""

    # --- 저장소 (Phase 2) ---
    database_url: str = ""
    """PostgreSQL 접속 URL. 비우면 인메모리 저장소로 뜬다(개발·데모용).

    예: postgresql://voiceguard:voiceguard@127.0.0.1:54321/voiceguard
    """

    # --- 판정 (Phase 2) ---
    match_threshold: float = 0.3767
    """원시 코사인 판정 임계값 (AS-Norm 미적용 시 폴백 경로에서 사용).

    WeSpeaker ResNet34-LM 기준, LibriSpeech dev-clean 40화자 2,400트라이얼의
    EER 지점 (EER 1.42%).

    참고: 캘리브레이션 이전 관례값 0.25는 같은 트라이얼에서 **FAR 8.75%**를 냈다 —
    인증 시스템에서 100번 중 8~9번 타인을 통과시키는 수준이다.
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
    옅어진다. WeSpeaker 기준 K=200에서만 EER 1.25%가 나왔다(다른 K는 1.42%).

    minDCF만 보면 K=300(0.0283)이 K=200(0.0392)보다 낫다. 오수락 억제를 더
    중시하는 배포에서는 K=300을 검토할 만하다 — 그 경우 EER 지점 임계값은 2.8136,
    minDCF 지점은 3.8811이다.
    """

    asnorm_threshold: float = 2.9673
    """AS-Norm 정규화 점수의 판정 임계값 (K=200의 EER 지점, EER 1.25%).

    정규화 점수는 코호트 기준 표준화 값이라 원시 코사인과 척도가 완전히 다르다
    (코사인처럼 [-1,1]에 갇히지 않는다). 두 임계값을 서로 바꿔 쓰면 안 된다.

    더 보수적인 운영(오수락 억제 우선)을 원하면 K=200의 minDCF 지점인 4.3525,
    또는 K=300 + 임계값 3.8811(minDCF 0.0283) 조합을 검토한다.
    """

    enroll_replaces_existing: bool = True
    """등록 시 기존 성문을 비활성화할지 여부.

    기본값 True는 "마지막 등록이 유효한 성문"이라는 단순한 모델이다. False로
    두면 여러 발화가 누적되고 검증은 그중 최대 유사도로 판정한다.
    """

    # --- 관리자 API (Phase 4) ---
    admin_token: str = ""
    """관리자 대시보드 API 인증 토큰.

    비어 있으면 관리자 엔드포인트가 **503으로 거부**된다. 감사 로그와 사용자
    ID가 노출되는 경로이므로, 토큰을 빠뜨린 배포가 조용히 공개되는 것보다
    아예 막히는 편이 안전하다.
    """

    # --- 런타임 ---
    warmup_on_startup: bool = True
    """기동 시 모델을 미리 적재해 첫 요청 지연을 없앤다."""


@lru_cache
def get_settings() -> Settings:
    return Settings()
