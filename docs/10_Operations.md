# 10_Operations.md
# 서비스 실행 가이드

세 구성요소를 순서대로 띄운다. **서버가 없으면 앱도 대시보드도 아무것도 못 한다** —
Thin Client 구조라 모든 분석이 서버에 있다.

```
[Flutter 앱]  ──HTTP──▶  [FastAPI 서버]  ──▶  [PostgreSQL + pgvector]
                              ▲
[Next.js 대시보드] ──HTTP─────┘
```

---

## 0. 사전 준비 (최초 1회)

### 필요한 것

| 항목 | 버전 | 비고 |
| :--- | :--- | :--- |
| Python | 3.10+ | 서버 |
| Node.js | 20+ | 대시보드 |
| Flutter | 3.27+ | 앱 (선택) |
| Docker | — | 개발용 PostgreSQL |

### 서버 환경 구성

```bash
cd server

# 가상환경 (python3-venv가 없으면 virtualenv 사용)
python3 -m virtualenv .venv

# PyTorch는 CPU 휠을 별도 인덱스에서 받는다 (GPU 휠은 2GB+)
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.5.1 torchaudio==2.5.1
.venv/bin/pip install -r requirements.txt
```

### 데이터베이스

```bash
docker run -d --name voiceguard-dev-db \
  -e POSTGRES_USER=voiceguard \
  -e POSTGRES_PASSWORD=voiceguard \
  -e POSTGRES_DB=voiceguard \
  -p 127.0.0.1:54321:5432 \
  pgvector/pgvector:pg16

# 마이그레이션 (순서대로)
for f in server/migrations/*.sql; do
  docker exec -i voiceguard-dev-db psql -U voiceguard -d voiceguard < "$f"
done
```

> 운영에서는 Supabase 등 관리형 PostgreSQL을 쓰고 `VG_DATABASE_URL`만 바꾼다.
> 스키마는 그대로 적용된다.

### 모델 가중치

임베딩·VAD 모델은 **첫 기동 시 자동으로 내려받는다**(약 100MB). 딥페이크 탐지만
수동으로 받아야 한다:

```bash
curl -L -o server/.model_cache/AASIST-L.pth --create-dirs \
  https://github.com/clovaai/aasist/raw/main/models/weights/AASIST-L.pth
```

### AS-Norm 코호트 (판정 품질에 필수)

코호트가 없으면 서버가 원시 코사인으로 폴백해 **판정 신뢰도가 크게 떨어진다.**
평가 데이터를 받아 코호트를 적재한다:

```bash
cd server && mkdir -p .data && cd .data
curl -O https://www.openslr.org/resources/12/test-clean.tar.gz
tar xzf test-clean.tar.gz && cd ..

VG_DATABASE_URL="postgresql://voiceguard:voiceguard@127.0.0.1:54321/voiceguard" \
  .venv/bin/python -m eval.seed_cohort --replace
```

> 코호트는 **실사용자와 무관한 화자**여야 한다. 사용자 성문이 섞이면 정규화가
> 자기 자신을 참조하게 된다 (02 §4.2).

---

## 1. 서버 실행

```bash
cd server

export VG_DATABASE_URL="postgresql://voiceguard:voiceguard@127.0.0.1:54321/voiceguard"
export VG_ADMIN_TOKEN="$(openssl rand -hex 32)"   # 대시보드용, 운영에선 반드시 무작위

./run.sh                                          # http://localhost:8000
```

확인:

```bash
curl -s localhost:8000/api/v1/health | python3 -m json.tool
```

```json
{
  "status": "ok",
  "models_loaded": true,
  "storage": "postgres",
  "storage_ok": true,
  "asnorm_active": true,      ← false면 코호트 미적재 (판정 신뢰도 낮음)
  "cohort_size": 310,
  "embedding_backend": "wespeaker"
}
```

API 문서는 `http://localhost:8000/docs`.

### 선택 기능

기본 비활성이며, **필요와 근거가 있을 때만** 켠다.

```bash
# 다중 화자 분리 — 혼합 오디오가 실제로 들어오는 배포에서만
export VG_SEPARATION_ENABLED=true      # CPU RTF 1.2, 6초 오디오에 ~7초

# 딥페이크 탐지 — 보안이 필요한 배포에서는 반드시 켤 것
export VG_ANTISPOOF_ENABLED=true       # ⚠ 임계값 재캘리브레이션 필수 (아래 §4)

# 음성 향상 — 실측에서 EER이 악화되어 권장하지 않음
export VG_ENHANCE_ENABLED=true
```

---

## 2. 관리자 대시보드 실행

```bash
cd web
npm install                            # 최초 1회

cat > .env.local <<EOF
VG_API_BASE_URL=http://127.0.0.1:8000
VG_ADMIN_TOKEN=<서버와 동일한 토큰>
EOF

npm run dev                            # http://localhost:3000
```

| 화면 | 용도 |
| :--- | :--- |
| `/` | 요청량, 응답 시간, 판정 분포, 점수 히스토그램 |
| `/speakers` | 등록 현황, **재등록 필요 사용자** |
| `/attempts` | 오딧 트레일 (원시·정규화·스푸핑 점수) |
| `/spoofing` | 스푸핑 차단 이력, 급증 경보 |
| `/calibration` | EER·minDCF, 임계값 영향도 |

> 대시보드는 **PostgreSQL 저장소에서만** 동작한다. 인메모리로 뜬 서버는 통계를
> 제공하지 않는다.

---

## 3. 앱 실행

```bash
cd app
flutter pub get                        # 최초 1회
flutter run --dart-define=VG_API_BASE_URL=http://localhost:8000
```

**Android 에뮬레이터에서는 반드시 `http://10.0.2.2:8000`을 쓴다** — 에뮬레이터의
`localhost`는 에뮬레이터 자신이다.

```bash
flutter run -d <device> --dart-define=VG_API_BASE_URL=http://10.0.2.2:8000
```

앱 없이 API만 시험하려면:

```bash
curl -X POST localhost:8000/api/v1/enroll -F "user_id=alice" -F "file=@voice.wav"
curl -X POST localhost:8000/api/v1/verify -F "user_id=alice" -F "file=@voice.wav"
```

---

## 4. 배포 전 반드시 할 것

### 임계값 재캘리브레이션

기본 임계값은 **LibriSpeech 기준**이다. 실제 서비스의 마이크·코덱·환경이 다르면
그대로 쓰면 안 된다.

```bash
cd server

# 화자 검증 임계값 — EER·minDCF 산출
.venv/bin/python -m eval.calibrate

# 딥페이크 탐지 임계값 — 길이별 오탐 측정 (⚠ 특히 중요)
.venv/bin/python -m eval.antispoof_length_eval
```

**딥페이크 탐지는 도메인에 매우 민감하다.** AASIST-L은 ASVspoof(VCTK)로 학습돼
다른 녹음 조건의 진짜 음성을 위조로 오인한다 — 구간별 오탐률이 1.0%에서 13.3%로
뛴다. 기본 임계값 0.999는 그 때문에 보수적으로 잡은 값이며, **자사 오디오로
정상 차단 1% 지점을 찾아 다시 설정해야 한다.**

### 보안 점검

| 항목 | 확인 |
| :--- | :--- |
| `VG_ADMIN_TOKEN` | 무작위 값인가 (`openssl rand -hex 32`) |
| 관리자 API | 외부에 노출되지 않는가 — 감사 로그·사용자 ID가 나간다 |
| `VG_ANTISPOOF_ENABLED` | 보안 배포라면 `true`인가 |
| DB 자격증명 | 웹 티어에 두지 않았는가 (대시보드는 토큰만 안다) |
| HTTPS | 성문·오디오가 평문으로 흐르지 않는가 |

### 성능 튜닝

```bash
.venv/bin/python -m eval.bench --url http://localhost:8000 --user-id <등록된 ID>
```

`VG_MAX_CONCURRENT_INFERENCE`(기본 4)는 24코어 기준 포화 지점이다. 코어 수가
다르면 위 벤치마크로 다시 측정해 정한다.

---

## 5. 문제 해결

| 증상 | 원인·조치 |
| :--- | :--- |
| `/health`의 `storage`가 `memory` | `VG_DATABASE_URL` 미설정. 재시작하면 등록이 사라진다 |
| `asnorm_active: false` | 코호트 미적재 → `python -m eval.seed_cohort --replace` |
| 검증이 `model_mismatch`로 거부 | 임베딩 모델을 바꿨다. 해당 사용자 재등록 필요 |
| 관리자 API가 503 | `VG_ADMIN_TOKEN` 미설정 (의도된 동작 — 열어두지 않는다) |
| 대시보드가 503 | 서버가 인메모리로 떴다. PostgreSQL로 재기동 |
| 정상 사용자가 자꾸 차단됨 | 딥페이크 탐지 임계값 문제 → §4의 재캘리브레이션 |
| 첫 요청이 매우 느림 | 모델 다운로드 중. `VG_WARMUP_ON_STARTUP=true`(기본)면 기동 시 적재된다 |

### 로그에서 볼 것

```
INFO app.main: 추론 동시 실행 상한: 4
INFO app.main: 모델 워밍업 완료
INFO app.db.session: AS-Norm 코호트 적재 완료: 310개 (top_k=200)   ← 없으면 정규화 꺼짐
WARNING app.db.session: 임포스터 코호트가 비어 있습니다             ← 판정 신뢰도 낮음
```

---

## 6. 정리

```bash
# 서버·대시보드 종료: Ctrl+C

# 개발 DB 제거 (등록·코호트 데이터가 함께 사라진다)
docker rm -f voiceguard-dev-db

# 평가 데이터 (재캘리브레이션에 필요하면 남겨둔다)
rm -rf server/.data
```
