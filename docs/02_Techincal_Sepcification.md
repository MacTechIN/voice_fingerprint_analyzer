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
| 오프라인 고정밀 (포렌식급) | SepFormer / MossFormer | WSJ0-2mix SOTA(SI-SNRi 22.3dB)급 정확도. 단, 어텐션의 2차 복잡도로 자원 소모 큼 |
| 실시간·장시간 스트리밍 | Mamba-TasNet (SSM) / RE-SepFormer | 시퀀스 길이에 선형 복잡도, 메모리 효율 |
| 에지·저대역 백엔드 | Codecformer-EL | 신경 오디오 코덱 임베딩 도메인 분리로 연산 2배+ 절감 |

*   **긴 오디오 처리:** 연속 음성 분리(CSS) 슬라이딩 윈도우 기법 적용. 윈도우 겹침 구간 유사도로 화자 추적 (긴 침묵 시 추적 단절 한계 인지).
*   **대안 설계 (권장):** 등록된 타겟 화자의 레퍼런스 임베딩을 분리망에 조건으로 주입하는 **타겟 화자 추출(TSE, Target Speaker Extraction)** 방식 — 순열 문제를 원천 회피하고 비타겟 화자·노이즈를 억압하여 검증 모듈 부담 경감.

## 3. 서버 사이드: 딥러닝 화자 인증 모델

### 3.1 임베딩 백본
*   **프로토타입 (Phase A):** ECAPA-TDNN (SpeechBrain, 192차원 임베딩) 또는 Resemblyzer(256차원). ECAPA-TDNN은 1D Res2Net + SE 채널 어텐션 기반의 사실상 표준 베이스라인으로, 노이즈에 강건하고 개방형 집합(Open-set) 성능 우수.
*   **확장 (Phase C 이후 교체):** **ERes2NetV2** (3D-Speaker) — 음성 분리 출력물이 2~3초 미만으로 분절되는 **단기 발화(Short-duration) 성능 저하 문제**를 다중 스케일 국소/전역 특징 융합으로 방어 (VoxCeleb1-O EER 0.61%). 에지·대규모 화자 환경에는 저연산 CAM++, 다국어 환경에는 Whisper-PMFA를 대안으로 검토.
*   **추론 엔진:** PyTorch 기반. 추후 트래픽 증가 시 NVIDIA Triton Inference Server 도입으로 동시 처리량(Throughput) 극대화.

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
*   **아키텍처:** **AASIST** (스펙트로-템포럴 그래프 어텐션) 기반. 고도화 옵션으로 SSL 프론트엔드 결합형 PT-SSL-AASIST(Wav2Vec 2.0 XLS-R + 웨이블릿 프롬프트 튜닝, EER ~3.58%) 또는 KAN 적용 AASIST3.
*   **배치 위치:** 검증 파이프라인 초입에 병렬 배치 — 임베딩 추출 이전에 합성 음성(LA 공격)을 1차 차단하고 `spoof_score`를 감사 로그에 기록.
*   **지속 검증:** Speech DF Arena 등 다중 데이터셋 벤치마크로 EER·F1 정기 점검.

## 6. 백엔드 및 인프라 사양
*   **클라우드 호스팅:** Google Cloud Platform (GCP) Cloud Run 또는 Compute Engine (사용자 선호 및 확장성 고려).
*   **Database:** Supabase (PostgreSQL + pgvector 확장) - 사용자 메타데이터 및 임베딩 벡터 관리. 벡터 차원은 채택 모델에 종속 (ECAPA-TDNN/ERes2NetV2: 192, Resemblyzer: 256) — 스키마에 차원·모델 버전 컬럼을 함께 기록하여 모델 교체 시 재등록(Re-enrollment) 마이그레이션 지원.
*   **코호트 테이블:** AS-Norm용 임포스터 임베딩 별도 테이블 (사용자 DB와 격리).
*   **오케스트레이션 툴킷:** WeSpeaker(AS-Norm·PLDA 내장) 및 3D-Speaker(온라인 특징 추출) 오픈소스 인프라 활용, 프로토타이핑은 SpeechBrain.
