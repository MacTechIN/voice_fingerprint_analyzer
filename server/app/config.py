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

    # --- 딥페이크 탐지 (Phase 8) ---
    antispoof_enabled: bool = False
    """합성 음성(TTS·보이스 컨버전) 탐지 적용 여부 (FR-16).

    기본 비활성이나, **보안이 필요한 배포에서는 반드시 켠다.** 잘 만든 합성
    음성은 화자 인증을 그대로 통과하므로 — 그것이 공격의 목적이다 — 별도
    방어가 없으면 성문만으로는 막을 수 없다.

    기본값을 false로 둔 이유는 가중치 파일이 저장소에 없어(라이선스는 MIT지만
    바이너리를 커밋하지 않는다) 내려받지 않은 환경에서 기동이 실패하지
    않게 하기 위해서다.
    """

    antispoof_weights: str = "./.model_cache/AASIST-L.pth"
    """AASIST-L 가중치 경로. clovaai/aasist(MIT)에서 받는다."""

    antispoof_threshold: float = 0.7
    """spoof 확률 판정 임계값. 이 값 이상이면 차단한다.

    ASVspoof 2019 LA 검증 세트 2,000개(bonafide 1,000 / spoof 1,000) 실측 운영 지점:

        임계값   위조 탐지   정상 차단
        0.50      99.9%      1.20%
        0.70      99.9%      1.00%   ← 채택
        0.90      99.4%      0.90%
        0.95      99.2%      0.70%
        0.99      99.0%      0.60%

    0.7을 택한 이유: 0.5와 탐지율이 같으면서(99.9%) 정상 사용자 차단은 적다 —
    0.5를 지배한다. 더 높이면 차단은 줄지만 위조를 놓치기 시작한다.

    **경고 — 긴 오디오에서는 오탐이 늘 수 있다.** 구간별 최댓값으로 집계하므로
    구간이 많을수록 하나라도 높게 나올 확률이 커진다. 평가 세트의 평균 길이는
    3.5초(대부분 1구간)라 이 위험을 충분히 재지 못했다. 실제로 12초짜리 LibriSpeech
    발화에서 최댓값이 0.79까지 올라간 사례가 있다. 배포 전 **자사 오디오 길이 분포로
    재측정**해야 한다.
    """

    # --- 음성 분리 (Phase 7) ---
    separation_enabled: bool = False
    """다중 화자 분리 적용 여부.

    **기본값이 비활성인 이유:** 분리는 공짜가 아니다. 단일 화자 오디오에
    적용하면 아티팩트만 더하고(02 §4.3 임베딩 왜곡), 추론 비용도 크다.
    다중 화자가 실제로 섞여 들어오는 배포에서만 켠다.

    켜면 검증 경로에서만 동작한다 — 등록은 통제된 환경에서 단일 화자로 받는
    것이 전제이므로 분리할 이유가 없다.
    """

    separation_model: str = "speechbrain/sepformer-whamr16k"
    """분리 모델. 16kHz라 리샘플링 왕복이 없고 잡음·잔향 조건으로 학습됐다."""

    separation_min_margin: float = 0.0
    """타겟 선택 1등과 2등의 최소 유사도 차이.

    차이가 이보다 작으면 어느 출력이 타겟인지 모호하다는 뜻이므로 경고를
    남긴다. 0이면 검사하지 않는다.
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

    torch_num_threads: int = 0
    """PyTorch 연산 스레드 수. 0이면 기본값(코어 수)을 그대로 둔다.

    **왜 명시가 필요한가:** 추론은 `anyio.to_thread`의 워커 스레드에서 도는데,
    그 안에서 PyTorch의 OpenMP 병렬 영역이 제대로 확장되지 않아 사실상 1스레드로
    동작하는 경우가 있다. 실측(6초 오디오, SepFormer 분리): 1스레드 28.7초 /
    4스레드 10.9초 / 12스레드 7.4초. 서버에서 30초가 나오던 것이 이 때문이었다.

    기동 시 한 번 설정하며 프로세스 전체에 적용된다. 동시 요청이 많은 배포에서는
    요청 간에 코어를 나눠 쓰도록 줄이는 편이 나을 수 있다.
    """

    max_concurrent_inference: int = 4
    """동시에 실행할 추론 요청 수 상한. 0이면 제한하지 않는다.

    **왜 제한하는가:** 추론은 CPU 바운드라 동시 실행을 늘려도 처리량이 비례해
    늘지 않는다. 실측(5초 오디오, 24코어):

        동시  1:  5.98 req/s, p50  142ms
        동시  2:  8.50 req/s, p50  224ms
        동시  4: 11.56 req/s, p50  330ms   ← 처리량 포화
        동시  8: 11.39 req/s, p50  698ms   ← 지연만 2배
        동시 16: 12.07 req/s, p50 1267ms   ← 지연만 4배

    4를 넘으면 처리량은 그대로인데 지연만 늘어난다. 상한을 두면 초과분이
    **대기열에서 기다리므로**, 모든 요청이 함께 느려지는 대신 처리 중인 요청은
    빠르게 끝난다 — 꼬리 지연이 예측 가능해진다.

    코어 수가 다른 환경에서는 `eval/bench.py`로 포화 지점을 다시 재서 정한다.
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
