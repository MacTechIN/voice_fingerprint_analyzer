# VoiceGuard Verification — AI 서버 (Phase 1)

성문(Voiceprint) 추출 API 서버. 오디오를 받아 **VAD로 무음을 제거**하고
**ECAPA-TDNN으로 192차원 화자 임베딩**을 반환한다.

> Phase 1 범위는 임베딩 추출까지다. 등록(`/enroll`)·검증(`/verify`)과 벡터 DB는
> Phase 2에서 붙는다 ([06_Development_Plan.md](../docs/06_Development_Plan.md)).

## 빠른 시작

```bash
cd server
python3 -m virtualenv .venv                                    # 최초 1회
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchaudio==2.5.1
.venv/bin/pip install -r requirements.txt
./run.sh                                                       # http://localhost:8000
```

첫 기동 시 사전학습 모델을 `.model_cache/`로 내려받는다(약 80MB). 이후에는 캐시를 쓴다.
API 문서는 `http://localhost:8000/docs`.

## 파이프라인

```
업로드 → 디코딩(16kHz/Mono 정규화) → Silero VAD(무음 제거) → ECAPA-TDNN → 192차원 임베딩
                                          │
                                          └─ 유효 발화 < 1.5초 → 422 반려
```

## API

### `GET /api/v1/health`

```json
{ "status": "ok", "model": "speechbrain/spkrec-ecapa-voxceleb", "models_loaded": true }
```

### `POST /api/v1/extract`

`multipart/form-data`로 `file`(16kHz 16-bit Mono WAV)을 보낸다.

```bash
curl -X POST http://localhost:8000/api/v1/extract -F "file=@speech.wav"
```

**200 성공**

```json
{
  "status": "success",
  "audio": {
    "duration_sec": 7.0,
    "speech_duration_sec": 3.068,
    "speech_ratio": 0.438,
    "sample_rate": 16000,
    "source_sample_rate": 16000,
    "source_channels": 1,
    "segments": [{ "start": 1.986, "end": 5.054 }]
  },
  "embedding": {
    "vector": [-0.1791, 0.067, "…(192개)"],
    "dim": 192,
    "model": "speechbrain/spkrec-ecapa-voxceleb",
    "l2_normalized": true
  },
  "elapsed_ms": 359.2
}
```

임베딩은 **L2 정규화**되어 있으므로 이후 유사도는 내적만으로 얻는다.
`model`·`dim`은 벡터와 **반드시 함께 저장**한다 — 모델 교체 시 재등록 대상을
식별하는 유일한 근거다.

**422 반려** — 서버 장애가 아니라 입력 문제다. `code`로 재녹음 안내를 분기한다.

| code | 의미 |
| :--- | :--- |
| `empty_file` | 업로드 본문이 비어 있음 |
| `file_too_large` | 허용 크기 초과 (기본 32MB) |
| `unreadable_audio` | 디코딩 실패 — 손상·미지원 포맷 |
| `audio_too_long` | 허용 길이 초과 (기본 300초) |
| `no_speech_detected` | VAD가 발화를 찾지 못함 |
| `speech_too_short` | 유효 발화가 하한 미달 (기본 1.5초) |

```json
{ "status": "error", "code": "speech_too_short",
  "detail": "유효 발화가 너무 짧습니다 (0.4초). 1.5초 이상 말해주세요." }
```

> 응답은 확장 가능 스키마다. Phase B~D에서 `normalized_score`(AS-Norm),
> `spoof_score`(딥페이크 탐지)가 최상위에 추가되므로, **클라이언트는 알 수 없는
> 필드를 무시하도록** 구현해야 한다 (Tolerant Reader).

## 설정

환경변수 접두사는 `VG_`. `.env` 파일도 읽는다.

| 변수 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `VG_MIN_SPEECH_SEC` | `1.5` | 유효 발화 길이 하한 |
| `VG_VAD_THRESHOLD` | `0.5` | Silero VAD 발화 확률 임계값 |
| `VG_MAX_AUDIO_SEC` | `300` | 단일 요청 최대 오디오 길이 |
| `VG_MAX_UPLOAD_BYTES` | `33554432` | 업로드 최대 크기 (32MB) |
| `VG_EMBEDDING_MODEL` | `speechbrain/spkrec-ecapa-voxceleb` | 임베딩 백본 |
| `VG_WARMUP_ON_STARTUP` | `true` | 기동 시 모델 선적재 |

## 테스트

```bash
.venv/bin/python -m pytest -q      # 22 passed
```

모델을 모킹하지 않고 실제 Silero VAD·ECAPA-TDNN을 통과시킨다. 임베딩 품질과
VAD 동작이 이 파이프라인의 본질이므로, 모킹하면 정작 검증할 것이 빠진다.

## 알려진 제약

- **의존성 고정 2건**: `huggingface-hub`는 1.x에서 `use_auth_token`을 제거했으나
  speechbrain 1.0.2가 아직 사용하므로 `0.26.5`로 고정했다. `requests`는 speechbrain이
  런타임에 쓰면서 의존성으로 선언하지 않아 직접 추가했다. speechbrain 업그레이드 시 재검토.
- **리샘플링은 선형 보간**이다. 규격(16kHz)을 지키는 정규 경로에서는 호출되지
  않는 예외 처리용이다. 규격 외 입력이 임베딩 품질에 영향을 준다면 클라이언트를
  고치는 것이 맞다.
- **상태 없는 단일 요청 처리**만 지원한다. 긴 오디오의 청크·세션 처리는
  [02 §2.4](../docs/02_Techincal_Sepcification.md)의 Phase 7 대상이다.

## 다음 단계 (Phase 2)

Supabase pgvector 연동, `POST /enroll`·`POST /verify` 추가, 코사인 유사도 판정.
