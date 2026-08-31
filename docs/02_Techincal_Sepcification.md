# 02_Technical_Specifications.md
# 기술 사양서 (Technical Specifications)

## 1. 클라이언트-서버 통신 규격
*   **네트워크 프로토콜:** RESTful API (HTTP/2 멀티파트 폼 데이터 전송) 또는 지연 시간 최소화를 위한 gRPC 양방향 스트리밍.
*   **페이로드 (Payload):**
    *   Client -> Server: 16kHz, 16-bit Mono WAV 오디오 파일 (또는 Base64 인코딩 스트림), 사용자 ID.
    *   Server -> Client: `{"status": "success", "match_probability": 92.5, "is_verified": true}`
    *   확장 응답 필드 (Phase B~D): `raw_cosine`(원시 유사도), `normalized_score`(AS-Norm 적용 점수), `spoof_score`(딥페이크 탐지 점수), `speech_duration`(VAD 통과 유효 발화 길이).

## 2. 서버 사이드: 전처리 및 음성 분리 프론트엔드

### 2.1 신호 모델
단일 채널 혼합 신호는 $C$명의 화자 직접 경로 신호 $s(c,t)$, 실내 잔향 반사음 $h(c,t)$, 배경 소음 $n(t)$의 합으로 모델링된다:

$$x(t) = \sum_{c=1}^{C}\bigl(s(c,t) + h(c,t)\bigr) + n(t)$$

서버 파이프라인의 목표는 $x(t)$로부터 타겟 화자의 파형 $\hat{s}$를 추정하여 임베딩 모듈에 공급하는 것이다.

### 2.2 전처리 (Pre-processing)
*   **VAD (Voice Activity Detection):** 파이프라인 최전단에 **Silero VAD** (경량, 저지연) 배치. 무음·단순 소음 구간을 제거하지 않으면 노이즈 임베딩이 성문 벡터를 오염시키므로 필수. 유효 발화 길이가 최소 기준(예: 1.5초) 미만이면 재녹음 요청 응답 반환.
*   **음성 향상 (Speech Enhancement, Phase B):** **DeepFilterNet** (PyTorch/ONNX, 8~48kHz 지원, CPU 실시간 가능) 결합으로 배경 소음의 임베딩 오염을 1차 차단. 인지 품질 지표는 PESQ, STOI로 검증.
*   **특징 추출:** 서버의 GPU/CPU 리소스를 활용하여 고속으로 80차원 Mel-filterbank 생성.

### 2.3 음성 분리 (Speech Separation, Phase C)
마스크 기반 인코더-마스커-디코더 구조를 채택하며, 훈련은 순열 문제(Permutation Problem) 해결을 위한 PIT(Permutation Invariant Training)와 SI-SNR 목적 함수를 전제로 한 사전학습 모델을 사용한다:

$$\text{SI-SNR} = 10\log_{10}\frac{\lVert\alpha s\rVert^2}{\lVert\hat{s}-\alpha s\rVert^2},\qquad \alpha=\frac{\langle\hat{s},s\rangle}{\lVert s\rVert^2}$$

**모델 선정 기준:**
| 시나리오 | 권장 모델 | 근거 |
| :--- | :--- | :--- |
| **기본 채택 (Phase 7)** | **MossFormer2-SS** (ClearerVoice-Studio, Apache-2.0) | 분리·향상·TSE를 한 툴킷에서 제공해 통합 비용 최소. 16kHz 사전학습 모델이 본 시스템 샘플레이트와 일치 |
| 프레임워크 통일 시 대안 | SepFormer (SpeechBrain) | 화자 검증(ECAPA-TDNN)까지 동일 프레임워크로 묶을 수 있음. WSJ0-2mix SI-SNRi 22.3dB |
| 에지·저대역 백엔드 | Codecformer-EL | 신경 오디오 코덱 임베딩 도메인 분리로 연산 2배+ 절감 |
| ~~실시간 선형 복잡도~~ | ~~Mamba-TasNet / SPMamba~~ | **채택 보류** — 공개 구현이 `mamba-ssm`의 CUDA 커널에 의존하여 **CPU 추론 불가**하고 연구코드 수준. 서버가 GPU 상시 확보 시 재평가 |

> **선정 근거:** 상세 실측 비교(스타·라이선스·최근 활동·CPU 추론 가능성)는 [08_OpenSource_Survey.md](08_OpenSource_Survey.md) 3장 참조.

*   **긴 오디오 처리:** 연속 음성 분리(CSS) 슬라이딩 윈도우 기법 적용. 상세 설계는 아래 2.4절 참조.
*   **대안 설계 (권장):** 등록된 타겟 화자의 레퍼런스 임베딩을 분리망에 조건으로 주입하는 **타겟 화자 추출(TSE, Target Speaker Extraction)** 방식 — 순열 문제를 원천 회피하고 비타겟 화자·노이즈를 억압하여 검증 모듈 부담 경감. 구현체로 `wenet-e2e/wesep`(BSRNN/TF-GridNet + WeSpeaker 임베딩 joint training)이 개념적으로 가장 부합하나, **저장소에 라이선스 파일이 없어 채택 전 확인이 필수**다.

### 2.4 청크 기반 연속 처리 설계 (CSS Chunking)

고정 길이로 자를 수 없는 긴 오디오(회의·통화 녹취, 스트리밍 입력)는 슬라이딩 윈도우로 분할해 순차 처리한다. 분리망은 윈도우 단위로만 동작하므로, **윈도우 경계를 넘어 화자 정체성을 유지하는 스티칭(Stitching) 로직**이 파이프라인의 핵심 상태 관리 지점이 된다.

#### 2.4.1 청크 파라미터 (기본값, 튜닝 대상)
| 파라미터 | 기본값 | 근거 |
| :--- | :--- | :--- |
| 윈도우 길이 `win` | 2.4 s | 분리망이 문맥을 확보하는 최소 길이. 짧으면 분리 품질 저하, 길면 지연·메모리 증가 |
| 홉 길이 `hop` | 1.2 s | 윈도우의 50% |
| 겹침 `overlap` | 1.2 s (`win - hop`) | 스티칭에 쓸 공통 구간. **최소 0.8s 이상 확보 필요** — 이보다 짧으면 화자 정렬 신뢰도가 급락 |
| 최대 세션 길이 | 운영 정책 | 초과 시 세션 분할 및 결과 병합 |

#### 2.4.2 윈도우 간 화자 스티칭 절차
1. 윈도우 `k`와 `k+1`의 겹침 구간에서, 각 출력 채널의 파형에 대해 화자 임베딩을 추출한다.
2. 두 윈도우 출력 채널 간 **코사인 유사도 행렬**을 만들고, 헝가리안 알고리즘(또는 탐욕적 최대 매칭)으로 채널 순열을 결정한다.
3. 결정된 순열로 `k+1`의 채널을 재정렬한 뒤, 겹침 구간은 크로스페이드(cross-fade)로 이어 붙여 경계 클릭 노이즈를 제거한다.
4. 매칭 최고 유사도가 임계값 미만이면 **새 화자 트랙 생성**으로 처리한다(기존 트랙에 강제 병합하지 않는다).

#### 2.4.3 추적 단절(Tracking Break) 대응 — 필수 설계
CSS는 겹침 구간의 음향적 연속성에 의존하므로, **긴 침묵이 끼면 화자 추적이 끊긴다**. 이 한계는 알고리즘상 제거할 수 없으므로 다음 두 계층으로 보완한다.

*   **1차 (상대적 추적):** 침묵 구간 길이가 `silence_max`(기본 1.5 s)를 넘으면 트랙 연속성을 신뢰하지 않고 트랙을 끊는다. 침묵 판정은 VAD 결과를 그대로 사용한다.
*   **2차 (절대적 재식별):** 끊어진 각 트랙을 **등록 임베딩(Enrollment vector)과 직접 대조**하여 화자 신원을 복구한다. 즉 화자 정체성의 최종 근거는 윈도우 간 상대 유사도가 아니라 **등록 성문과의 절대 매칭**이며, 스티칭은 트랙을 길게 이어 임베딩 품질을 높이기 위한 보조 수단으로 한정한다. 이 원칙 덕분에 추적이 끊겨도 검증 결과 자체는 손상되지 않는다.
*   **TSE 적용 시:** 타겟 화자 추출 방식을 쓰면 출력 채널이 항상 1개(타겟)로 고정되므로 스티칭 순열 문제 자체가 소멸한다. **긴 오디오 처리에서 TSE가 CSS보다 구조적으로 유리한 결정적 이유**이며, Phase 7에서 TSE를 우선 검토하는 근거이기도 하다.

#### 2.4.4 서버 구현 요건
*   **상태 관리:** 청크 처리는 상태 유지(stateful)다. 세션 ID별로 직전 윈도우의 꼬리 버퍼(`overlap` 길이)와 활성 화자 트랙의 임베딩을 보관한다. FastAPI 워커가 다중화되므로 세션-워커 어피니티를 보장하거나 세션 상태를 외부 저장소(Redis 등)에 두어야 한다.
*   **백프레셔:** 입력 속도가 처리 속도를 초과하면 큐가 무한 증가하므로, 세션당 큐 상한과 초과 시 정책(가장 오래된 청크 폐기 또는 요청 거절)을 명시한다.
*   **지연 예산:** 스트리밍 응답의 이론적 최소 지연은 `win`(첫 윈도우 확보) + 분리·임베딩 추론 시간이다. `win=2.4s` 기준으로 실시간 응답 목표를 세울 때 이 하한을 반영한다.
*   **경계 정렬:** 청크 경계는 가급적 VAD가 검출한 발화 경계에 맞춘다. 발화 중간을 자르면 해당 청크의 임베딩 품질이 떨어진다.

> **⚠ 확인 필요 (Open Issue):** 본 설계는 CSS 표준 방식에 따른 것으로, **사내 서버 `BEF2.0`의 청크 처리 방식과의 정합성은 아직 확인되지 않았다.** BEF2.0의 청크 파라미터(윈도우/홉 길이), 세션 상태 관리 방식, 스트리밍 프로토콜 규격을 확보한 뒤 본 절의 파라미터와 인터페이스를 재정렬해야 한다. 상세는 [09_BEF2.0_Integration.md](09_BEF2.0_Integration.md) 참조.

## 3. 서버 사이드: 딥러닝 화자 인증 모델

### 3.1 임베딩 백본
*   **프로토타입 (Phase A):** ECAPA-TDNN (SpeechBrain, 192차원 임베딩). 1D Res2Net + SE 채널 어텐션 기반의 사실상 표준 베이스라인으로, 노이즈에 강건하고 개방형 집합(Open-set) 성능이 우수하며 `SpeakerRecognition.from_hparams()` 한 줄로 검증 데모 구성이 가능해 착수 속도가 가장 빠르다.
    *   ~~Resemblyzer~~는 **비권장** — 2023-10 이후 개발이 사실상 중단되었고 GE2E d-vector 구세대 정확도라 프로덕션 후보에서 제외한다.
*   **본구축 (Phase B 전환):** **WeSpeaker** (`wenet-e2e/wespeaker`, Apache-2.0) — 훈련부터 AS-Norm/PLDA/캘리브레이션(`bin/score_norm.py`), ONNX export, C++ onnxruntime 런타임, Triton GPU 서버까지 한 저장소에서 완결되는 유일한 프로덕션 지향 툴킷. SpeechBrain은 AS-Norm과 서빙이 없으므로 Phase 6 시점에 코어를 WeSpeaker로 전환한다.
*   **확장 (Phase C 이후 모델 교체):** **ERes2NetV2** (3D-Speaker) — 음성 분리 출력물이 2~3초 미만으로 분절되는 **단기 발화(Short-duration) 성능 저하 문제**를 다중 스케일 국소/전역 특징 융합으로 방어 (VoxCeleb1-O EER 0.61%). 에지·대규모 화자 환경에는 저연산 CAM++, 다국어 환경에는 Whisper-PMFA를 대안으로 검토. 단 3D-Speaker에는 AS-Norm 스크립트가 없으므로 **정규화는 WeSpeaker 구현을 이식**한다.
*   **추론 엔진:** PyTorch 기반. 배포 계층으로 `k2-fsa/sherpa-onnx`(WeSpeaker/3D-Speaker ONNX 모델을 다언어·websocket으로 서빙) 검토, 트래픽 증가 시 NVIDIA Triton Inference Server 도입으로 동시 처리량(Throughput) 극대화.

### 3.2 손실 함수 (파인튜닝 시 적용 기준)
사전학습 모델을 자체 데이터로 파인튜닝할 경우, 열린 집합 검증 성능을 위해 가산 각도 마진 손실(ArcFace/AAM)을 표준으로 채택한다. 임베딩과 클래스 중심 벡터를 $\ell_2$ 정규화해 스케일 $s$의 초구(Hypersphere)에 고정하고, 타겟 클래스 각도 $\theta_{y_i}$에 마진 $m$을 가산한다:

$$\mathcal{L}_{ArcFace} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{s\,\cos(\theta_{y_i}+m)}}{e^{s\,\cos(\theta_{y_i}+m)}+\sum_{j=1,\,j\neq y_i}^{C} e^{s\,\cos\theta_j}}$$

마진 $m$은 초구면 상의 측지선 거리(Geodesic distance)에 대응하며 클래스 내 밀집·클래스 간 분리를 강제한다. 대안으로 확률론적 마진의 Q-Margin($q_y=\exp(-s\cdot m)$, $\alpha$-divergence 기반), 발화 조건 간 일관성 확보용 AAMSupCon(지도 대조학습 + AAM) 검토 가능.

## 4. 서버 사이드: 스코어링 및 판정 백엔드

### 4.1 유사도 산출
등록 임베딩 $e$와 검증 임베딩 $t$ 간 **코사인 유사도**를 기본 스코어 $S(e,t)$로 사용.

### 4.2 AS-Norm 점수 정규화 (Phase B 필수)
원시 유사도에 고정 임계값을 적용하면 화자 상태·발화 길이·잔존 소음·분리 아티팩트로 인한 점수 편차 때문에 실패율이 급증한다. 사전 구축된 임포스터 코호트(Impostor Cohort)에서 $e$, $t$ 각각과 가장 혼동되기 쉬운 상위 $K$개 발화를 적응적으로 선택해 평균 $\mu_{cohort}$, 표준편차 $\sigma_{cohort}$를 계산하고 대칭 정규화한다:

$$S_{AS\text{-}Norm} = \frac{1}{2}\left(\frac{S(e,t)-\mu_{cohort}(e)}{\sigma_{cohort}(e)} + \frac{S(e,t)-\mu_{cohort}(t)}{\sigma_{cohort}(t)}\right)$$

*   **코호트 DB:** 실제 사용자와 무관한 화자 발화 임베딩 수백~수천 개를 서버에 사전 적재 (pgvector 별도 테이블).
*   **성능 지표:** EER(Equal Error Rate, FAR=FRR 교차점) 및 minDCF를 기준으로 임계값 캘리브레이션. 정규화 점수를 0~100% 일치 확률로 매핑해 클라이언트에 반환.

### 4.3 파이프라인 통합 유의사항: 임베딩 왜곡 (Embedding Distortion)
음성 분리 모델은 원본에 없는 비선형 아티팩트와 주파수 탈락을 발생시키는 반면, 화자 인식 모델은 깨끗한 단일 발화로 학습되어 있어 직렬 결합(Cascaded) 시 도메인 불일치로 검증 성능이 붕괴할 수 있다. 대응 전략:
1.  **탈동조화(Decoupled) 파인튜닝 (단기 권장):** 분리망 출력의 아티팩트·잔향을 인위적으로 첨가한 증강 데이터(AC-SIM 방식)로 임베딩 백본을 파인튜닝.
2.  **TSE 파라다임 도입:** 등록 임베딩 조건부 분리로 아티팩트 원인 자체를 축소.
3.  **결합 종단간 학습(EEND-SS, 장기):** 분리·식별·화자 수 카운팅을 단일 멀티태스크 프레임워크로 공동 최적화 (대규모 학습 인프라 확보 시).

## 5. 서버 사이드: 딥페이크 탐지 (Anti-Spoofing, Phase D)

**2단 캐스케이드 구성**을 채택한다 — 전체 요청에 무거운 SSL 모델을 적용하면 응답 지연 예산을 초과하기 때문이다.

| 단계 | 모델 | 라이선스 | 성능 | 적용 범위 |
| :--- | :--- | :--- | :--- | :--- |
| 1차 스크리닝 | **AASIST-L** (`clovaai/aasist`) | MIT | ASVspoof2019 LA EER 0.99% (85K 파라미터, CPU 실시간) | 전체 요청 |
| 2차 정밀 판정 | **XLSR + AASIST** (`TakHemlata/SSL_Anti-spoofing`) | MIT | ASVspoof2021 LA EER 0.82%, DF 2.85% | 1차 의심 샘플만 |

*   **배치 위치:** 검증 파이프라인 초입에 병렬 배치 — 임베딩 추출 이전에 합성 음성(LA 공격)을 1차 차단하고 `spoof_score`를 감사 로그에 기록.
*   **통합 유의사항:** SSL_Anti-spoofing 원본은 고정 커밋 fairseq / torch 1.8에 의존한다. 본 프로젝트는 **추론만 필요**하므로 프론트엔드를 HuggingFace `transformers`의 `wav2vec2-xls-r-300m`으로 교체하는 경량 포팅을 권장한다.
*   **라이선스 배제:** ~~AASIST3(KAN 적용)~~는 공개 구현(`lab260ru/AASIST3`)이 **CC BY-NC-ND 4.0**으로 상용 이용과 파생물 제작이 모두 금지되어 **채택 불가**하다. SpeechBrain에는 anti-spoofing 레시피가 존재하지 않으므로(recipes 전수 확인) 이 단계에서는 사용할 수 없다.
*   **지속 검증:** 채택 전 `MuSAELab/AUDDT`(33개 딥페이크 데이터셋 통합 평가)로 자사 오디오 조건(코덱·샘플레이트)의 교차 데이터셋 일반화를 검증하고, `asvspoof-challenge/2021` eval-package로 t-DCF/EER 표준 지표를 산출한다.

## 6. 백엔드 및 인프라 사양
*   **클라우드 호스팅:** Google Cloud Platform (GCP) Cloud Run 또는 Compute Engine (사용자 선호 및 확장성 고려).
*   **Database:** Supabase (PostgreSQL + pgvector 확장) - 사용자 메타데이터 및 임베딩 벡터 관리. 벡터 차원은 채택 모델에 종속 (ECAPA-TDNN/ERes2NetV2: 192) — 스키마에 차원·모델 버전 컬럼을 함께 기록하여 모델 교체 시 재등록(Re-enrollment) 마이그레이션 지원.
    *   **인덱스:** 화자 수 10만 미만 규모에서는 HNSW 인덱스 + 코사인 거리로 충분. 사용자 계정·인증 로그·임베딩을 단일 트랜잭션으로 다룰 수 있다는 점이 전용 벡터 DB(Qdrant/Milvus) 대비 결정적 이점이므로 pgvector를 유지하고, 수백만 화자 규모에 도달할 때만 재검토한다.
    *   **접근 라이브러리:** `pgvector-python` (asyncpg/SQLAlchemy 바인딩)으로 FastAPI 비동기 경로와 통합.
*   **코호트 테이블:** AS-Norm용 임포스터 임베딩 별도 테이블 (사용자 DB와 격리).
*   **오케스트레이션 툴킷:** WeSpeaker(AS-Norm·PLDA·캘리브레이션 내장, ONNX·Triton 서빙 완비) 및 3D-Speaker(ERes2NetV2·CAM++ 사전학습 공급, 온라인 특징 추출) 오픈소스 인프라 활용, 프로토타이핑은 SpeechBrain.
*   **참조 구현 부재 인지:** FastAPI + 화자검증 + pgvector를 한 번에 묶은 신뢰할 만한 오픈소스 저장소는 존재하지 않는다(조사 확인). SpeechBrain(임베딩) + pgvector-python(검색) + Supabase RPC 코사인 검색 패턴을 직접 조합해야 하므로 초기 개발 공수를 여기에 배정한다. `yeyupiaoling/VoiceprintRecognition-Pytorch`는 서버 코드를 포함하지 않으므로 학습·추론 파이프라인 참조용으로만 사용한다.
