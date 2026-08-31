# VoiceGuard Verification — AI 서버

성문(Voiceprint) 기반 화자 인증 API. 오디오를 받아 **VAD로 무음을 제거**하고
**ECAPA-TDNN으로 192차원 임베딩**을 만들어 **pgvector에 등록**하고,
**AS-Norm 정규화 점수로 1:1 검증**한다.

**실측 성능 (LibriSpeech dev-clean, 40화자 2,400트라이얼): EER 1.25%, minDCF 0.0317.**
측정 방법과 한계는 [eval/README.md](eval/README.md) 참조.

## 빠른 시작

```bash
cd server
python3 -m virtualenv .venv                                    # 최초 1회
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchaudio==2.5.1
.venv/bin/pip install -r requirements.txt

# DB (개발용 pgvector 컨테이너)
docker run -d --name voiceguard-dev-db \
  -e POSTGRES_USER=voiceguard -e POSTGRES_PASSWORD=voiceguard -e POSTGRES_DB=voiceguard \
  -p 127.0.0.1:54321:5432 pgvector/pgvector:pg16
docker exec -i voiceguard-dev-db psql -U voiceguard -d voiceguard < migrations/001_init.sql
docker exec -i voiceguard-dev-db psql -U voiceguard -d voiceguard < migrations/002_cohort.sql

export VG_DATABASE_URL="postgresql://voiceguard:voiceguard@127.0.0.1:54321/voiceguard"

# AS-Norm 코호트 적재 (없으면 원시 코사인으로 폴백한다)
.venv/bin/python -m eval.seed_cohort --replace                  # eval/README.md의 데이터 준비 선행

./run.sh                                                       # http://localhost:8000
```

`VG_DATABASE_URL`을 비우면 **인메모리 저장소**로 뜬다(개발·데모용). 프로세스를
재시작하면 등록된 성문이 사라지므로, 그 사실을 기동 로그와 `/health`의 `storage`
필드에 드러낸다.

첫 기동 시 사전학습 모델을 `.model_cache/`로 내려받는다(약 80MB).
API 문서는 `http://localhost:8000/docs`.

## 파이프라인

```
업로드 → 디코딩(16kHz/Mono) → Silero VAD(무음 제거) → ECAPA-TDNN → 192차원 임베딩
                                    │                                  │
                                    │                  ┌───────────────┴───────────────────┐
                        유효 발화 < 1.5초 → 422        │ enroll: pgvector 저장              │
                                                       │ verify: 코사인 → AS-Norm 정규화     │
                                                       │         → 임계값 판정               │
                                                       └───────────────────────────────────┘
```

AS-Norm은 임포스터 코호트를 기준으로 점수를 표준화해 화자 상태·발화 길이·잔존
소음에 따른 점수 편차를 흡수한다. 코호트가 비어 있으면 원시 코사인으로 폴백하되,
그 사실을 응답의 `scoring_method`와 `/health`의 `asnorm_active`에 드러낸다.

## API

### `GET /api/v1/health`

```json
{
  "status": "ok",
  "model": "speechbrain/spkrec-ecapa-voxceleb",
  "models_loaded": true,
  "storage": "postgres",
  "storage_ok": true,
  "asnorm_active": true,
  "cohort_size": 310
}
```

### `POST /api/v1/extract`

오디오에서 성문을 추출한다 (저장하지 않는다).

```bash
curl -X POST http://localhost:8000/api/v1/extract -F "file=@speech.wav"
```

### `POST /api/v1/enroll`

사용자 성문을 등록한다 (FR-04).

```bash
curl -X POST http://localhost:8000/api/v1/enroll -F "user_id=alice" -F "file=@speech.wav"
```

```json
{
  "status": "success",
  "user_id": "alice",
  "enrollment_id": 6,
  "replaced": 1,
  "audio": { "duration_sec": 7.0, "speech_duration_sec": 3.068, "…": "…" },
  "embedding": { "vector": ["…(192개)"], "dim": 192, "model": "…", "l2_normalized": true },
  "elapsed_ms": 415.0
}
```

`VG_ENROLL_REPLACES_EXISTING`(기본 `true`)이면 기존 성문을 비활성화하고 `replaced`에
그 건수를 담는다. **비활성화는 삭제가 아니다** — 재등록 후 문제가 생겼을 때 이전
벡터를 추적할 수 있어야 하고, 감사 로그가 참조하는 이력이 사라지면 안 된다.

### `POST /api/v1/verify`

업로드된 음성이 해당 사용자와 동일인인지 판정한다 (FR-05, FR-06).

```bash
curl -X POST http://localhost:8000/api/v1/verify -F "user_id=alice" -F "file=@speech.wav"
```

```json
{
  "status": "success",
  "user_id": "alice",
  "is_verified": true,
  "match_probability": 100.0,
  "raw_cosine": 0.7677,
  "normalized_score": 11.4606,
  "scoring_method": "as_norm",
  "threshold": 3.0033,
  "compared_enrollments": 1,
  "audio": { "…": "…" },
  "elapsed_ms": 299.8
}
```

`scoring_method`가 `as_norm`이면 `normalized_score`로 판정한 것이고, `raw_cosine`은
참고용이다. 코호트가 없어 폴백한 경우 `raw_cosine`으로 판정하며 `normalized_score`는
`null`이 된다.

> **정규화 점수와 원시 코사인은 임계값을 공유하지 않는다.** 정규화 점수는 코호트
> 표준편차 단위라 코사인처럼 [-1, 1]에 갇히지 않는다(실측 genuine 11.46). 두
> 임계값(`VG_MATCH_THRESHOLD` / `VG_ASNORM_THRESHOLD`)을 서로 바꿔 쓰면 안 된다.

> **`match_probability`는 확률이 아니다.** 판정 경계를 50%에 맞춘 구간 선형
> 재척도다. 진짜 확률로 쓰려면 로지스틱 캘리브레이션이 필요하다.

등록 발화가 여러 개면 **최대 유사도**로 판정한다(`compared_enrollments`에 대조
건수). 평균 벡터를 쓰지 않는 이유는 발화 조건이 다른 벡터들을 뭉개 어느 쪽과도
덜 닮은 중심이 만들어질 수 있어서다. 다중 등록 발화의 평균 성문은 01 §2의 향후
보강 후보로 남겨두었다.

### 반려 코드 (422)

서버 장애가 아니라 입력·상태 문제다. `code`로 재녹음/재등록 안내를 분기한다.

| code | 의미 |
| :--- | :--- |
| `empty_file` | 업로드 본문이 비어 있음 |
| `file_too_large` | 허용 크기 초과 (기본 32MB) |
| `unreadable_audio` | 디코딩 실패 — 손상·미지원 포맷 |
| `audio_too_long` | 허용 길이 초과 (기본 300초) |
| `no_speech_detected` | VAD가 발화를 찾지 못함 |
| `speech_too_short` | 유효 발화가 하한 미달 (기본 1.5초) |
| `not_enrolled` | 검증 대상 사용자의 등록 성문 없음 |
| `model_mismatch` | 등록 성문이 다른 모델 것 — 재등록 필요 |

`model_mismatch`는 조용히 넘기지 않는다. 모델이 다른 벡터끼리 비교하면 무의미한
유사도가 나오고, 그 값이 그대로 인증 판정에 쓰이기 때문이다 (02 §6).

> 응답은 확장 가능 스키마다. Phase B~D에서 `normalized_score`(AS-Norm),
> `spoof_score`(딥페이크 탐지)가 최상위에 추가되므로, **클라이언트는 알 수 없는
> 필드를 무시하도록** 구현해야 한다 (Tolerant Reader).

## 데이터 모델

`migrations/001_init.sql` 참조. Supabase에 그대로 적용 가능하다.

| 테이블 | 용도 |
| :--- | :--- |
| `speaker_enrollments` | 등록 성문. `embedding vector(192)` + `model`·`dim` 메타데이터 |
| `verification_attempts` | 검증 감사 로그. 원시 점수·정규화 점수·판정 결과·IP (03 오딧 트레일) |
| `impostor_cohort` | AS-Norm 임포스터 코호트. **사용자 성문과 물리적으로 분리** |

벡터와 함께 `model`·`dim`을 **반드시** 저장한다. 모델을 교체하면 기존 벡터와
호환되지 않는데, 메타데이터가 없으면 어떤 행이 어느 모델 것인지 사후에 알 수 없다.

코호트는 실제 사용자와 무관한 화자여야 한다. 사용자 성문이 섞이면 정규화가 자기
자신을 참조하게 되어 사칭자 통계가 오염된다.

### 코호트 적재

```bash
VG_DATABASE_URL="postgresql://..." .venv/bin/python -m eval.seed_cohort --replace
```

코호트 없이 뜨면 경고 로그와 함께 원시 코사인으로 폴백한다. 운영에서는 반드시
적재하고 `/health`의 `asnorm_active`로 확인한다.

## 설정

환경변수 접두사는 `VG_`. `.env` 파일도 읽는다.

| 변수 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `VG_DATABASE_URL` | (없음) | PostgreSQL URL. 비우면 인메모리 |
| `VG_MATCH_THRESHOLD` | `0.3333` | 원시 코사인 임계값 (AS-Norm 폴백 시, EER 지점) |
| `VG_ASNORM_ENABLED` | `true` | AS-Norm 정규화 사용 여부 |
| `VG_ASNORM_TOP_K` | `200` | 코호트 상위 K개 적응 선택 |
| `VG_ASNORM_THRESHOLD` | `3.0033` | AS-Norm 판정 임계값 (EER 지점) |
| `VG_ENROLL_REPLACES_EXISTING` | `true` | 등록 시 기존 성문 비활성화 |
| `VG_MIN_SPEECH_SEC` | `1.5` | 유효 발화 길이 하한 |
| `VG_VAD_THRESHOLD` | `0.5` | Silero VAD 발화 확률 임계값 |
| `VG_MAX_AUDIO_SEC` | `300` | 단일 요청 최대 오디오 길이 |
| `VG_MAX_UPLOAD_BYTES` | `33554432` | 업로드 최대 크기 (32MB) |
| `VG_EMBEDDING_MODEL` | `speechbrain/spkrec-ecapa-voxceleb` | 임베딩 백본 |
| `VG_WARMUP_ON_STARTUP` | `true` | 기동 시 모델 선적재 |

## 테스트

```bash
.venv/bin/python -m pytest -q                                   # 인메모리만 (67 passed, 2 skipped)

VG_TEST_DATABASE_URL="postgresql://voiceguard:voiceguard@127.0.0.1:54321/voiceguard" \
  .venv/bin/python -m pytest -q                                 # Postgres 포함 (69 passed)
```

모델을 모킹하지 않고 실제 Silero VAD·ECAPA-TDNN을 통과시킨다. 저장소는 인메모리와
Postgres 두 구현을 **같은 계약 테스트**로 검증해 둘이 어긋나지 않게 한다.

## 알려진 제약

- **임계값은 LibriSpeech 기준이다.** 조용한 환경의 또렷한 영어 낭독 음성으로
  측정했으므로, 실제 서비스의 잡음·짧은 발화·다국어 조건과 다르다. **출발점이지
  최종값이 아니며**, 운영 데이터가 쌓이면 그 분포로 재캘리브레이션해야 한다.
  화자 80명은 통계적으로 넉넉하지 않아 표본 오차도 있다.
- **음성 향상(DeepFilterNet)이 아직 없다.** Phase 6 Micro-process 2 미착수.
- **임베딩 코어가 아직 SpeechBrain이다.** WeSpeaker 전환(Micro-process 1) 미착수.
- **의존성 고정 2건**: `huggingface-hub`는 1.x에서 `use_auth_token`을 제거했으나
  speechbrain 1.0.2가 아직 사용하므로 `0.26.5`로 고정했다. `requests`는 speechbrain이
  런타임에 쓰면서 의존성으로 선언하지 않아 직접 추가했다.
- **리샘플링은 선형 보간**이다. 규격(16kHz)을 지키는 정규 경로에서는 호출되지 않는
  예외 처리용이다.
- **상태 없는 단일 요청 처리**만 지원한다. 긴 오디오의 청크·세션 처리는
  [02 §2.4](../docs/02_Techincal_Sepcification.md)의 Phase 7 대상이다.
- **1:N 식별은 없다.** 1:1 검증만 지원하며, `user_id`로 좁힌 뒤 소수의 벡터만
  비교하므로 ANN 인덱스도 두지 않았다. 1:N을 도입하면 HNSW(`vector_cosine_ops`)를
  추가하고 유사도 계산을 SQL(`<=>`)로 옮긴다.

## 다음 단계

- **Phase 6 잔여**: WeSpeaker 코어 전환, DeepFilterNet 음성 향상
- **Phase 3**: Flutter 클라이언트에서 녹음 → `/enroll`·`/verify` → 결과 시각화 E2E
