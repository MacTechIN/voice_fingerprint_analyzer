# 06_Development_Plan.md
# 개발 계획서 (Micro-process & Vertical Stackable)

앱 경량화를 목표로 하므로, **API 서버 우선 구축 후 클라이언트를 연동하는 방식**으로 스택을 쌓아 올립니다. Phase 1~5는 프로토타입(Phase A) 완성 경로이며, Phase 6~8은 연구 보고서(『음성 분리 및 화자 대조』)의 4단계 기술 로드맵 — ① 전처리·분리 프론트엔드, ② 초구면 잠재 공간 최적화, ③ 분리 왜곡 극복·AS-Norm 백엔드, ④ 보안 방어막 — 을 단계적으로 반영하는 확장 경로입니다.

## Phase 1: AI 서버 구축 및 API 설계 (Week 1-2) — ✅ 완료

*   *Micro-process:* Python(FastAPI) 기반으로 ECAPA-TDNN 모델 로드 환경 구성. 파이프라인 최전단에 Silero VAD 결합 (무음·소음 구간 제거 후 임베딩 추출).
*   *Output:* 오디오 파일을 업로드하면 임베딩 벡터(모델 기본 차원, ECAPA-TDNN 기준 192차원)를 반환하는 `POST /extract` API 완성. 유효 발화 길이 미달 시 사유 코드 반려 응답 포함.

### 구현 결과 (`server/`)

| 항목 | 결과 |
| :--- | :--- |
| 엔드포인트 | `POST /api/v1/extract`, `GET /api/v1/health` |
| 파이프라인 | 디코딩(16kHz/Mono 정규화) → Silero VAD → ECAPA-TDNN |
| 임베딩 | 192차원, L2 정규화, `model`·`dim` 메타데이터 동봉 |
| 반려 사유 코드 | 6종 (`empty_file`, `file_too_large`, `unreadable_audio`, `audio_too_long`, `no_speech_detected`, `speech_too_short`) |
| 테스트 | 22개 통과 (모델 모킹 없이 실측) |
| 실측 성능 | 7초 오디오 처리 359ms, 서버 기동 4초(모델 선적재 포함) |

**구현 중 확정한 설계 판단**

*   **반려는 5xx가 아니라 422다.** 발화 부족·무음은 서버 장애가 아니라 입력 문제이며, 클라이언트는 `code`별로 다른 재녹음 안내를 띄운다(04 §4). 사유 코드 값은 클라이언트 계약이므로 변경 금지 대상이다.
*   **응답을 `audio`/`embedding` 중첩 객체로 묶었다.** 최상위를 평평하게 두면 Phase B~D에서 `normalized_score`·`spoof_score`가 붙을 때 계약이 지저분해진다(01 §2 확장 여지).
*   **임베딩을 저장 전 L2 정규화한다.** 이후 코사인 유사도를 내적만으로 얻는다(Phase 2 스코어링 단순화).
*   **추론은 워커 스레드로 넘긴다.** CPU 바운드 작업이 GIL을 잡으면 이벤트 루프가 다른 요청의 I/O를 처리하지 못한다.
*   **규격 외 입력(44.1kHz 스테레오 등)도 서버가 흡수한다.** 단 리샘플링은 선형 보간이므로 품질이 중요하면 클라이언트를 고치는 것이 맞다.

**해결한 의존성 충돌 2건** (requirements.txt에 고정, speechbrain 업그레이드 시 재검토)

*   `huggingface-hub` 1.x가 `use_auth_token` 인자를 제거했으나 speechbrain 1.0.2가 아직 사용 → `0.26.5`로 고정.
*   `requests`를 speechbrain이 런타임에 쓰면서 의존성으로 선언하지 않음 → 직접 추가.

## Phase 2: Vector DB 통합 및 인증 로직 (Week 3-4) — ✅ 완료

*   *Micro-process:* Supabase pgvector 구축 (임베딩 모델명·버전·차원 메타데이터 컬럼 포함). 두 오디오 벡터를 비교해 % 확률을 도출하는 코사인 유사도 로직 작성.
*   *Vertical Slice:* 서버 단독으로 등록(`POST /enroll`) 및 검증(`POST /verify`) API 엔드투엔드 작동 확인.
*   *설계 원칙:* 응답 JSON은 확장 가능 스키마로 설계 (추후 `normalized_score`, `spoof_score` 등 필드 무중단 추가 대비).

### 구현 결과 (`server/`)

| 항목 | 결과 |
| :--- | :--- |
| 엔드포인트 추가 | `POST /api/v1/enroll`, `POST /api/v1/verify` |
| 스키마 | `speaker_enrollments`(vector(192) + model·dim 메타데이터), `verification_attempts`(감사 로그) |
| 저장소 | PostgreSQL 16 + pgvector, 인메모리 폴백 (동일 계약) |
| 반려 코드 추가 | `not_enrolled`, `model_mismatch` |
| 테스트 | 51개 통과 (Postgres 계약 테스트 포함) |
| 실측 성능 | 등록 415ms, 검증 273ms |

**구현 중 확정한 설계 판단**

*   **모델 불일치는 조용히 넘기지 않는다.** 등록 성문이 현재 임베딩 모델과 다르면 `model_mismatch`로 거부한다. 차원만 같고 모델이 다른 벡터를 비교하면 무의미한 유사도가 나오는데, 그 값이 그대로 인증 판정에 쓰이기 때문이다(02 §6).
*   **재등록은 삭제가 아니라 비활성화다.** 재등록 후 문제가 생겼을 때 이전 벡터를 추적할 수 있어야 하고, 감사 로그가 참조하는 이력이 사라지면 안 된다.
*   **다중 등록 시 평균이 아니라 최대 유사도로 판정한다.** 평균 벡터는 발화 조건이 다른 벡터들을 뭉개 어느 쪽과도 덜 닮은 중심을 만들 수 있다. 평균 성문은 01 §2의 향후 보강 후보로 유지.
*   **유사도 계산을 SQL이 아닌 애플리케이션에서 한다.** 1:1 검증은 사용자당 소수 벡터만 비교하므로 SQL로 넘길 이득이 없고, Phase 6에 AS-Norm이 들어오면 어차피 스코어 후처리를 애플리케이션에서 해야 한다. 1:N 식별 도입 시 `<=>`와 HNSW로 옮긴다.
*   **`match_probability`는 확률이 아님을 문서·코드에 명시했다.** 판정 경계를 50%에 맞춘 구간 선형 재척도이며, `verification_attempts.normalized_score`는 Phase 6 자리로 비워 두었다(가짜 값을 채우지 않음).
*   **저장소는 실패 시 기동을 중단한다.** 모델 워밍업과 달리 저장소 없이는 등록·검증이 성립하지 않으므로, 반쯤 동작하는 서버를 띄우는 것보다 즉시 실패하는 편이 낫다.

### ⚠ Phase 6까지 유효한 경고

임계값 `0.25`는 **SpeechBrain 관례값일 뿐 본 시스템 데이터로 캘리브레이션한 값이 아니다.** E2E 검증에서 음향적으로 다른 합성 음원이 코사인 0.394로 임계값을 넘어 `is_verified=true`가 나왔다. 합성음이라 화자 인식 성능 자체를 재는 상황은 아니지만, **고정 임계값 판정의 취약성을 그대로 보여준다**(02 §4.2가 예고한 문제). 실제 운영 임계값은 Genuine/Impostor 분포를 모아 EER·minDCF로 결정해야 하며, 그전까지 판정 결과를 신뢰해서는 안 된다.

## Phase 3: 경량화된 Cross-Platform App 개발 (Week 5-7)
*   *Micro-process:* Flutter 프로젝트 세팅, 마이크 제어 및 오디오 파일화 로직 구현 (16kHz/Mono 캡처, 최소 3초 발화 강제, 입력 레벨 피드백).
*   *Vertical Slice:* 앱 UI에서 버튼을 눌러 녹음 후, Phase 2에서 만든 API로 전송하여 화면에 00% 확률 결과를 출력하는 E2E 통신 완성.

## Phase 4: Web Dashboard 구축 (Week 8)
*   *Micro-process:* Next.js로 관리자 웹 구축, API 서버의 DB(Supabase)에 접근하여 통계 로그 구현.

## Phase 5: 서버 최적화 및 네트워크 지연 최소화 (Week 9-10)
*   *Micro-process:* 클라이언트 체감 속도 향상을 위해 오디오 전송 압축, FastAPI 워커(Worker) 수 조정, 네트워크 응답 지연(Latency) 개선. (필요시 gRPC 통신으로 업그레이드).
*   *성능 개선 참조 자료:* 장시간 오디오 처리 비용·지연이 병목으로 확인되면 [chunked-preprocess-pattern.md](chunked-preprocess-pattern.md)(SEP v2/LEP v2 검증 패턴)를 검토한다. 클라이언트 측 겹침 분할 + 조각별 선처리 결과 재사용으로 재분석 비용을 제거하는 방식이며, 현행 02 §2.4 설계를 대체하는 것이 아니라 그 바깥 계층으로 얹는 구조를 상정한다. **현 시점에는 도입하지 않고 보관한다.**

---

## Phase 6: 판정 신뢰성 강화 — 코어 전환 + 음성 향상 + AS-Norm (확장 Phase B)

> **진행 상황:** Micro-process 3~4(AS-Norm + 캘리브레이션) **✅ 완료**.
> Micro-process 1(WeSpeaker 코어 전환)과 2(DeepFilterNet)는 미착수.
> Phase 2에서 드러난 미캘리브레이션 임계값 문제를 먼저 해소하기 위해 순서를 조정했다.

### ✅ 캘리브레이션 실측 결과 (2026-08-31)

**데이터:** LibriSpeech (CC BY 4.0). 평가는 dev-clean 40화자 2,400트라이얼
(genuine 1,200 / impostor 1,200), 코호트는 test-clean 40화자 310개.
**두 split의 화자는 겹치지 않는다** — 코호트에 평가 화자가 섞이면 정규화가 자기
자신을 참조해 측정이 무의미해진다.

| 방식 | EER | 임계값 | minDCF | 분리도 |
| :--- | ---: | ---: | ---: | ---: |
| 원시 코사인 | 1.67% | 0.3333 | 0.0617 | 4.80 |
| AS-Norm (K=50) | 1.33% | 2.3335 | 0.0408 | 3.74 |
| AS-Norm (K=100) | 1.25% | 2.7029 | 0.0383 | 3.90 |
| **AS-Norm (K=200)** ← 채택 | **1.25%** | **3.0033** | **0.0317** | 4.36 |
| AS-Norm (K=300) | 1.25% | 2.8819 | 0.0325 | 4.63 |

**AS-Norm 적용으로 EER 1.67% → 1.25% (25% 개선), minDCF 0.0617 → 0.0317 (49% 개선).**

K 선정: EER이 100/200/300에서 동률이었고, 그중 minDCF가 가장 낮은 K=200을 택했다.
minDCF는 사칭 시도가 드문 실제 환경을 반영해 FAR에 더 큰 가중을 두므로 인증
시스템에 더 적합한 기준이다.

### ⚠ Phase 2 경고가 실측으로 확인됨

Phase 2에서 "임계값 0.25는 미캘리브레이션"이라고 경고한 것이 수치로 확인됐다.
같은 트라이얼에서 **임계값 0.25는 FAR 6.67%, FRR 0.75%**를 냈다 — 인증 시스템에서
**100번 중 6~7번 타인을 통과시키는** 수준이다. 캘리브레이션 후 기본값을 갱신했다.

| 설정 | 이전 | 이후 |
| :--- | :--- | :--- |
| `VG_MATCH_THRESHOLD` (원시 코사인 폴백) | 0.25 (관례값) | **0.3333** (EER 지점) |
| `VG_ASNORM_TOP_K` | — | **200** |
| `VG_ASNORM_THRESHOLD` | — | **3.0033** (EER 지점) |

### 구현 결과

| 항목 | 결과 |
| :--- | :--- |
| AS-Norm | `app/services/asnorm.py` — 대칭 정규화, 코호트 상위 K 적응 선택 |
| 코호트 저장소 | `impostor_cohort` 테이블 (사용자 성문과 물리적 분리), 기동 시 메모리 적재 |
| 평가 하네스 | `eval/` — 트라이얼 생성, 임베딩 캐시, DET/EER/minDCF, 캘리브레이션 |
| 응답 확장 | `normalized_score`, `scoring_method` 필드 추가 (무중단) |
| 감사 로그 | `normalized_score` 컬럼 실제 기록 시작 (Phase 2에서는 NULL이었음) |
| 테스트 | 69개 통과 (AS-Norm 단위 11 + 통합 7 신규) |
| E2E 실측 | 실제 LibriSpeech 화자로 동일인 통과(정규화 11.46) / 타인 2건 거부(-2.22, -3.00) |

**구현 중 확정한 설계 판단**

*   **정규화 점수와 원시 코사인은 임계값을 공유하지 않는다.** 정규화 점수는 코호트
    표준편차 단위라 코사인처럼 [-1,1]에 갇히지 않는다(실측 genuine 11.46). 두
    임계값을 별도 설정으로 분리하고 문서에 교차 사용 금지를 명시했다.
*   **코호트가 없으면 원시 코사인으로 폴백하되 그 사실을 드러낸다.** 응답의
    `scoring_method`와 `/health`의 `asnorm_active`로 노출한다. 정규화가 꺼진 채
    운영되는 것을 모르고 지나치는 것이 가장 위험하다.
*   **평가는 운영 경로와 같은 조건으로 한다.** `eval/extract.py`는 서버와 동일하게
    VAD를 적용하고, VAD가 반려했을 발화는 트라이얼에서도 제외한다.

### 남은 한계

*   **LibriSpeech는 낭독 음성이다.** 조용한 환경의 또렷한 영어 오디오북이며, 실제
    서비스의 잡음·짧은 발화·다국어 조건과 다르다. 이 임계값은 **출발점이지 최종값이
    아니며**, 운영 데이터가 쌓이면 그 분포로 재캘리브레이션해야 한다.
*   화자 80명은 통계적으로 넉넉하지 않아 임계값 추정치에 표본 오차가 있다.
*   코호트가 LibriSpeech 화자라 실사용자 집단과 음향 특성이 다르면 정규화 효과가
    측정값보다 작을 수 있다.

### 원 계획 (Micro-process 1·2는 미착수)

*   *Micro-process 1 (코어 전환):* 임베딩 엔진을 SpeechBrain → **WeSpeaker**로 전환. SpeechBrain에는 AS-Norm과 서빙 계층이 없어 Phase 6 요구사항을 충족할 수 없으므로, 이 시점이 전환 마감선이다. 전환 시 기존 등록 벡터의 재등록(Re-enrollment) 마이그레이션 수행.
*   *Micro-process 2:* DeepFilterNet(v3) 소음 억제를 VAD 다음 단계에 결합. PESQ/STOI로 향상 전후 품질 비교 검증. 생성형 향상 모델(resemble-enhance 등)은 성문 왜곡 위험 때문에 배제.
*   *Micro-process 3:* 임포스터 코호트 임베딩 DB 구축(공개 데이터셋 활용, 사용자 DB와 격리) 후 WeSpeaker `bin/score_norm.py`를 활용해 AS-Norm 대칭 정규화 스코어링 구현 (자체 구현 대신 검증된 구현체 이식).
*   *Micro-process 4:* 자체 평가 트라이얼 셋 구성 → EER·minDCF 측정 스크립트 작성 → 웹 대시보드에 임계값 캘리브레이션 UI 추가.
*   *Vertical Slice:* 동일 검증 요청에 대해 원시 점수와 정규화 점수를 병기 반환·로그하고, 정규화 적용 전후 EER 개선을 수치로 확인.

## Phase 7: 다중 화자 대응 — 음성 분리 / TSE (확장 Phase C)
*   *Micro-process 1:* **ClearerVoice-Studio(MossFormer2-SS, 16kHz)** 사전학습 모델로 2인 혼합 오디오 분리 PoC. 프레임워크를 SpeechBrain으로 통일하는 편이 유리하다고 판단되면 SepFormer로 대체. Mamba 계열(SPMamba 등)은 CUDA 커널 의존으로 CPU 추론이 불가하므로 GPU 상시 확보 전까지 후보에서 제외.
*   *Micro-process 2:* 등록 임베딩을 조건으로 하는 타겟 화자 추출(TSE) 방식 실험 — 순열 문제 회피 및 파이프라인 단순화 비교. 후보 `wenet-e2e/wesep`은 **착수 전 라이선스 확인이 선행 조건**(저장소에 LICENSE 파일 없음).
*   *Micro-process 3:* **임베딩 왜곡 대응:** 분리망 출력 아티팩트를 첨가한 증강 데이터로 임베딩 백본 파인튜닝(탈동조화 전략). 이 시점에 백본을 단기 발화에 강한 ERes2NetV2(3D-Speaker)로 교체 평가 (교체 시 기존 사용자 재등록 마이그레이션 절차 수행).
*   *Vertical Slice:* 2인 동시 발화 녹음 → 분리 → 타겟 화자 검증 성공률을 단일 화자 대비 벤치마크.

## Phase 8: 보안 방어막 — Anti-Spoofing (확장 Phase D)
*   *Micro-process 1:* **AASIST-L**(85K 파라미터, CPU 실시간) 사전학습 모델을 검증 파이프라인 초입에 배치, `spoof_score` 임계 기반 1차 차단 로직 구현.
*   *Micro-process 2:* 1차 의심 샘플에 한해 **XLSR + AASIST**(`SSL_Anti-spoofing`) 2차 정밀 판정을 적용하는 캐스케이드 구성. 원본의 fairseq/torch 1.8 고정 의존을 피하기 위해 프론트엔드를 HuggingFace `wav2vec2-xls-r-300m`으로 교체 포팅 (추론 전용이므로 포팅 범위 작음).
*   *Micro-process 3:* TTS 생성 음성으로 자체 공격 시뮬레이션 → 탐지율(EER·F1) 측정. 교차 데이터셋 일반화는 `AUDDT`(33개 데이터셋), 표준 지표 산출은 `asvspoof-challenge/2021` eval-package 사용. **AASIST3는 CC BY-NC-ND 라이선스로 상용 배치가 불가하므로 업그레이드 경로에서 제외.**
*   *Vertical Slice:* 합성 음성 인증 시도가 차단되고 관리자 대시보드에 스푸핑 로그·경보가 표시되는 E2E 확인.

## 리스크 및 대응 원칙
| 리스크 | 대응 |
| :--- | :--- |
| 짧은 발화(2~3초)에서 임베딩 신뢰도 저하 | 클라이언트 최소 발화 강제(FR-01) + Phase 7에서 ERes2NetV2 전환 |
| 분리 아티팩트로 인한 검증 성능 붕괴 | 아티팩트 증강 파인튜닝(탈동조화) 우선, EEND-SS 결합 학습은 장기 과제로 보류 |
| 원시 점수 고정 임계값의 높은 실패율 | Phase 6 AS-Norm을 필수 경로로 편성 |
| 모델 교체 시 기존 성문 벡터 비호환 | 벡터에 모델 버전 메타데이터 저장, 재등록 마이그레이션 플로우 사전 설계 (Phase 6 WeSpeaker 전환 시 최초 적용) |
| **완성형 참조 저장소 부재** | FastAPI+화자검증+pgvector를 묶은 오픈소스가 없음(조사 확인) → Phase 1~2에 직접 조합 공수를 별도 배정 |
| **오픈소스 라이선스 리스크** | AASIST3(CC BY-NC-ND, 상용 불가)·wesep(LICENSE 부재)·record(루트 LICENSE 부재) 등은 채택 전 법무 확인 절차를 게이트로 설정 |
| **CPU 추론 제약** | Mamba 계열 배제, 스푸핑 탐지 캐스케이드 구성 등이 이 제약에서 도출됨. GPU 예산 확보 시 모델 선택 재평가 |
