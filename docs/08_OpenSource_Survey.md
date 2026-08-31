# 08_OpenSource_Survey.md
# 성문 분석 오픈소스 전수 조사 보고서 (멀티에이전트 병렬 조사)

- **조사일:** 2026-08-31
- **조사 방법:** 4개 리서치 에이전트를 병렬 가동하여 GitHub REST API(`gh api`)로 스타 수·라이선스·최근 push·저장소 구조를 **실측**하고, README 원문 및 웹 검색으로 교차 검증. 표의 수치는 조사 시점 API 응답값 기준.
- **조사 범주:** ① 화자 임베딩/인증 툴킷 ② 음성 분리·향상·VAD ③ 딥페이크 탐지(Anti-Spoofing) ④ 서빙 인프라·벡터 DB·클라이언트

---

## 1. 최종 채택 스택 (Executive Summary)

| 파이프라인 단계 | 채택 | 라이선스 | 도입 Phase | 비고 |
| :--- | :--- | :--- | :---: | :--- |
| VAD | **snakers4/silero-vad** | MIT | 1 | ONNX 동봉, 30ms 청크당 ~1ms(CPU 1스레드) |
| 임베딩 (프로토타입) | **speechbrain** (ECAPA-TDNN) | Apache-2.0 | 1 | 한 줄로 검증 데모 가능, 최속 시작점 |
| 임베딩 (본구축) | **wenet-e2e/wespeaker** | Apache-2.0 | 6 | 훈련→AS-Norm→ONNX→Triton 서빙 완결 |
| 임베딩 모델 공급 | **modelscope/3D-Speaker** (ERes2NetV2/CAM++) | Apache-2.0 | 7 | 단기 발화 최고 성능군 사전학습 모델 |
| 스코어 정규화 | **wespeaker** `bin/score_norm.py` | Apache-2.0 | 6 | AS-Norm·PLDA·calibration 내장 |
| 음성 향상 | **Rikorose/DeepFilterNet** v3 | MIT/Apache-2.0 | 6 | CPU RTF ≈0.04, 판별형이라 성문 왜곡 위험 낮음 |
| 음성 분리 | **modelscope/ClearerVoice-Studio** (MossFormer2-SS) | Apache-2.0 | 7 | 분리·향상·TSE 통합 툴킷, 16kHz 모델 |
| 타겟 화자 추출(TSE) | wenet-e2e/wesep *(조건부)* | **없음 ⚠** | 7 | 라이선스 파일 부재 — 채택 전 확인 필수 |
| 딥페이크 탐지 (1차) | **clovaai/aasist** AASIST-L | MIT | 8 | 85K 파라미터, EER 0.99%, CPU 실시간 |
| 딥페이크 탐지 (2차) | **TakHemlata/SSL_Anti-spoofing** | MIT | 8 | XLSR+AASIST, ASVspoof2021 LA EER 0.82% |
| 벡터 DB | **pgvector** + pgvector-python | PostgreSQL License / MIT | 2 | HNSW + 코사인, 화자 <10만 규모 충분 |
| 클라이언트 | **record** + **dio** + **audio_waveforms** | BSD-3 / MIT / MIT | 3 | record는 루트 LICENSE 부재, pub.dev 표기 확인 |
| 배포 계층 (옵션) | **k2-fsa/sherpa-onnx** | Apache-2.0 | 5~ | WeSpeaker/3D-Speaker ONNX 모델 다언어 서빙 |
| 대규모 서빙 (옵션) | **triton-inference-server** | BSD-3 | 트래픽 증가 시 | 초기 MVP에는 과함 |

### 1.1 조사로 정정된 기존 문서의 오류
| 항목 | 기존 기술 | 조사 결과 | 조치 |
| :--- | :--- | :--- | :--- |
| Resemblyzer | 프로토타입 임베딩 선택지 | 2023-10 개발 중단, GE2E 구세대 정확도 | **비권장으로 강등** |
| Mamba-TasNet | "실시간·스트리밍용 권장" | `mamba-ssm`이 CUDA 의존 → **CPU 추론 불가**, 연구코드 수준 | 채택 보류, 추적 관찰 |
| AASIST3 | "고도화 옵션" | **CC BY-NC-ND 4.0** — 상용·파생 금지 | **배제** |
| yeyupiaoling 저장소 | "RESTful API 서버 배포 설계" | 저장소 구조 확인 결과 **서버 코드 미포함** (README의 DIY 가이드뿐) | 학습·추론 파이프라인 참조용으로 격하 |
| SpeechBrain Anti-Spoofing | 암묵적 기대 | recipes 44종 전수 확인, **ASVspoof 레시피 없음** | 명시적 제외 |
| 음성 분리 1순위 | SepFormer | MossFormer2-SS가 최신 SOTA + 통합 툴킷 | ClearerVoice-Studio로 교체 |

---

## 2. 화자 임베딩 / 화자 인증 툴킷

| 저장소 | ⭐ | 라이선스 | 최근 활동 | 지원 모델 | 사전학습 | AS-Norm | ONNX/서빙 |
| :--- | ---: | :--- | :--- | :--- | :---: | :---: | :--- |
| [wenet-e2e/wespeaker](https://github.com/wenet-e2e/wespeaker) | 1,395 | Apache-2.0 | 2026-07 | ECAPA-TDNN, CAM++, ERes2Net, ResNet, RepVGG, ReDimNet, Whisper-PMFA | O | **O** | **최상급** (ONNX·onnxruntime·MNN·Triton) |
| [modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker) | 3,126 | Apache-2.0 | 2025-12 | CAM++, **ERes2NetV2**, ERes2Net, ECAPA-TDNN, RDINO/SDPN | O | X | ONNX export + 런타임 |
| [speechbrain/speechbrain](https://github.com/speechbrain/speechbrain) | 11,797 | Apache-2.0 | 2026-08 (v1.1.1) | ECAPA-TDNN, x-vector, ResNet-TDNN | O (HF) | X | 자체 서빙 없음 |
| [NVIDIA-NeMo/Speech](https://github.com/NVIDIA/NeMo) | 18,361 | Apache-2.0 | 2026-08 (v3.0.0) | TitaNet-L/S, ECAPA-TDNN, SpeakerNet | O | 제한적 | ONNX + Riva |
| [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio) | 10,487 | MIT | 2026-08 (4.0.7) | 임베딩은 **WeSpeaker ResNet34 포팅** | O (HF gated) | X | X |
| [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | 14,498 | Apache-2.0 | 2026-08 | (학습 없음) WeSpeaker/3D-Speaker ONNX 실행 | 모델 zoo | X | **ONNX 전용 런타임, 12개 언어** |
| [resemble-ai/Resemblyzer](https://github.com/resemble-ai/Resemblyzer) | 3,300 | Apache-2.0 | 2023-10 (중단) | GE2E d-vector 256차원 | O | X | X |
| [yeyupiaoling/VoiceprintRecognition-Pytorch](https://github.com/yeyupiaoling/VoiceprintRecognition-Pytorch) | 1,312 | Apache-2.0 | 2025-12 | EcapaTdnn, CAM++, ERes2Net 등 | O | X | 서버 코드 없음 |
| [Snowdar/asv-subtools](https://github.com/Snowdar/asv-subtools) | 636 | Apache-2.0 | 2024-08 | x-vector, ECAPA, ResNet (Kaldi 하이브리드) | 일부 | **O** | ONNX 예제 |
| [clovaai/voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer) | 1,175 | MIT | 2026-04 | ResNetSE34, VGGVox + metric learning loss | 베이스라인 | X | X |

**결론:** 1위 **wespeaker**(유일한 "훈련→정규화→서빙" 완결 툴킷, pyannote 4.x조차 임베딩을 여기서 가져다 쓰는 사실상 표준), 2위 **3D-Speaker**(ERes2NetV2 원저작·최고 성능 사전학습, 단 AS-Norm 스크립트 이식 필요), 3위 **speechbrain**(Phase 1 최속 프로토타입). `astorfi/3D-convolutional-speaker-recognition`, `philipperemy/deep-speaker`, `HarryVolek/PyTorch_Speaker_Verification` 등 검색 상위 구현은 2019~2021년 이후 방치되어 후보에서 제외.

---

## 3. 음성 분리 · 향상 · VAD

| 저장소 | ⭐ | 라이선스 | 최근 push | 모델/기법 | 사전학습 | 실시간 | CPU | ONNX |
| :--- | ---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | 10,088 | MIT | 2026-08 | Silero VAD v5/v6 | O (~2MB) | **O** | **O** | **O** |
| [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | 4,642 | MIT+Apache-2.0 | 2024-10 (정체) | DFN 1/2/3, 48kHz 풀밴드 | O | **O** | **O** (RTF≈0.04) | **O** |
| [modelscope/ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) | 4,455 | Apache-2.0 | 2025-08 | MossFormer2 (SE/SS), FRCRN, AV-MossFormer2-TSE | O | X | △ | X |
| [speechbrain/speechbrain](https://github.com/speechbrain/speechbrain) | 11,797 | Apache-2.0 | 2026-08 | SepFormer, MetricGAN+, CRDNN VAD | O | △ | △ | △ |
| [asteroid-team/asteroid](https://github.com/asteroid-team/asteroid) | 2,584 | MIT | 2026-05 | Conv-TasNet, DPRNN, DPTNet, SuDoRM-RF | O | X | O | △ |
| [wenet-e2e/wesep](https://github.com/wenet-e2e/wesep) | 311 | **없음 ⚠** | 2025-10 | **TSE 전문**: BSRNN, TF-GridNet + wespeaker 임베딩 joint training | O | △ | △ | △ |
| [TEN-framework/ten-vad](https://github.com/TEN-framework/ten-vad) | 2,251 | Apache-2.0 **+추가조건** | 2026-02 | 프레임 단위 저지연 VAD | O | **O** | **O** | **O** |
| [JusperLee/SPMamba](https://github.com/JusperLee/SPMamba) | 226 | Apache-2.0 | 2024-12 | Mamba 기반 분리 (TF-GridNet + Bi-Mamba) | O | X | **X (CUDA 의존)** | X |
| [RoyChao19477/SEMamba](https://github.com/RoyChao19477/SEMamba) | 274 | MIT | 2025-12 | Mamba 기반 향상 | O | X | **X (CUDA 의존)** | X |
| [JusperLee/TIGER](https://github.com/JusperLee/TIGER) | 441 | MIT | 2026-04 | 경량 고효율 분리 | O | △ | O | X |
| [facebookresearch/demucs](https://github.com/facebookresearch/demucs) | 10,359 | MIT | **아카이브(2024-04)** | HTDemucs (음악 소스 분리) | O | △ | △ | X |
| [resemble-ai/resemble-enhance](https://github.com/resemble-ai/resemble-enhance) | 2,401 | MIT | 2024-12 | 생성형 denoise+enhance | O | X | X | X |
| [timsainb/noisereduce](https://github.com/timsainb/noisereduce) | 1,874 | MIT | 2025-08 | 스펙트럴 게이팅 (비DNN) | — | O | O | — |

**단계별 결론**
- **VAD:** `silero-vad` 1순위. TEN-VAD는 성능 우위를 주장하나 Apache-2.0에 **추가 조건이 붙은 변형 라이선스**라 상용 전 법무 검토 필요. pyannote segmentation-3.0은 정확도 최상이나 배치 전용 + HF gated이므로 오프라인 등록 경로에만 보조 사용.
- **음성 향상:** `DeepFilterNet` v3. **판별형(discriminative) 모델이라 생성형(resemble-enhance) 대비 성문 왜곡 위험이 낮다는 점이 화자 인증에서 결정적**. 커밋 정체(2024-10)는 성숙 단계로 판단.
- **음성 분리:** `ClearerVoice-Studio`(MossFormer2-SS) 1순위 — 분리·향상·TSE 통합으로 파이프라인 비용 최소, 16kHz 모델이 우리 샘플레이트와 일치. 시스템 전체를 SpeechBrain으로 통일할 경우 SepFormer가 역전 가능.
- **TSE:** `wesep`이 개념적으로 정확히 부합(등록 임베딩을 단서로 사용)하나 **라이선스 파일 부재**가 채택 전제 조건.
- **제외:** demucs(아카이브 + 음악 특화), Mamba 계열(CPU 추론 불가).

---

## 4. 딥페이크 탐지 / Anti-Spoofing

| 저장소 | ⭐ | 라이선스 | 최근 push | 아키텍처 | 사전학습 | 성능(README) | 통합 난이도 |
| :--- | ---: | :--- | :--- | :--- | :---: | :--- | :--- |
| [clovaai/aasist](https://github.com/clovaai/aasist) | 295 | MIT | 2023-06 | AASIST / AASIST-L (raw waveform + GAT) | O | ASVspoof2019 LA EER **0.83%**, AASIST-L 0.99%(85K params) | **낮음** |
| [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing) | 179 | MIT | 2023-09 | **XLSR-300M(SSL) + AASIST** | O | ASVspoof2021 LA **0.82%**, DF **2.85%** | 중~상 (고정 fairseq) |
| [asvspoof-challenge/2021](https://github.com/asvspoof-challenge/2021) | 256 | 미표기 | 2024-06 | 공식 베이스라인 4종 + **eval-package**(t-DCF/EER) | 부분 | 기준선 | 중 |
| [asvspoof-challenge/asvspoof5](https://github.com/asvspoof-challenge/asvspoof5) | 76 | 미표기 | 2024-09 | RawNet2, AASIST + SASV 트랙 | X | 평가 프로토콜 | 중 |
| [nii-yamagishilab/project-NN-Pytorch-scripts](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts) | 363 | BSD-3 | **2026-07 (활발)** | LFCC-LCNN, SSL 프론트엔드 CM | 일부 | 논문별 상이 | 중 |
| [lab260ru/AASIST3](https://github.com/lab260ru/AASIST3) | 17 | **CC BY-NC-ND 4.0 ⚠** | 2026-05 | Wav2Vec2 + KAN + GAT | O (HF) | 미기재 | 낮음 (**상용 불가**) |
| [QiShanZhang/SLSforASVspoof-2021-DF](https://github.com/QiShanZhang/SLSforASVspoof-2021-DF) | 73 | 미표기 | 2025-02 | XLS-R + SLS 분류기 (ACM MM'24) | O | DF ~1.9% | 중~상 |
| [piotrkawa/deepfake-whisper-features](https://github.com/piotrkawa/deepfake-whisper-features) | 117 | MIT | 2025-04 | Whisper 인코더 + SpecRNet/LCNN | O | In-The-Wild 평가 | 중 |
| [MuSAELab/AUDDT](https://github.com/MuSAELab/AUDDT) | 36 | Research-Use-Only | 2026-05 | **33개 딥페이크 데이터셋 통합 평가 툴킷** | — | 평가 도구 | 낮음 |
| [eurecom-asp/rawnet2-antispoofing](https://github.com/eurecom-asp/rawnet2-antispoofing) | 70 | MIT | 2023-08 | RawNet2 원저자 구현 | X | ICASSP'21 재현 | 중 |
| [yzyouzhang/AIR-ASVspoof](https://github.com/yzyouzhang/AIR-ASVspoof) | 140 | MIT | — | one-class 학습 | — | — | 중 |
| [LetterLiGo/SafeEar](https://github.com/LetterLiGo/SafeEar) | 190 | 커스텀 | — | 프라이버시 보존형 탐지 (CCS'24) | — | — | 중 |

**결론:** **2단 캐스케이드 구성**을 권고한다.
1. **1차 스크리닝:** `AASIST-L` (85K 파라미터, EER 0.99%, CPU 실시간) — 전체 요청에 적용.
2. **2차 정밀 판정:** `SSL_Anti-spoofing` (XLSR+AASIST, LA EER 0.82%) — 의심 샘플만. fairseq/torch 1.8 고정 의존이 걸림돌이므로, 추론 전용인 우리 용도에서는 HuggingFace `transformers`의 `wav2vec2-xls-r-300m`으로 프론트엔드를 교체하는 경량 포팅 권장.
3. **검증 인프라:** 채택 전 `AUDDT`(33개 데이터셋)로 자사 오디오 조건(코덱·샘플레이트)의 교차 데이터셋 일반화를 검증하고, `asvspoof-challenge/2021` eval-package로 t-DCF/EER 표준 산출.

**라이선스 주의:** MIT로 안전한 것은 aasist, SSL_Anti-spoofing, RawGAT-ST, AIR-ASVspoof, deepfake-whisper-features. **AASIST3(CC BY-NC-ND)는 상용 배치 불가**, asvspoof-challenge 공식 저장소와 SLS는 라이선스 미표기라 법무 검토 필요.

---

## 5. 서빙 인프라 · 벡터 DB · 클라이언트

### 5.1 모델 서빙
| 저장소 | ⭐ | 라이선스 | 최근 활동 | 평가 |
| :--- | ---: | :--- | :--- | :--- |
| [triton-inference-server/server](https://github.com/triton-inference-server/server) | 10,951 | BSD-3 | 2026-08 | 동적 배칭·멀티프레임워크. 초기 MVP엔 과함, 트래픽 증가 시 업그레이드 경로 |
| [microsoft/onnxruntime-inference-examples](https://github.com/microsoft/onnxruntime-inference-examples) | 1,679 | MIT | 2026-08 | ECAPA-TDNN ONNX 서빙 시 직접 참조 |
| [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) | 12,671 | Apache-2.0 | 2026-08 | `paddlespeech server`가 화자 임베딩 HTTP 서비스 기성 제공 — **설계 참조용**(Paddle 종속이라 채택 비권장) |
| [Mrkomiljon/DeepVoiceGuard](https://github.com/Mrkomiljon/DeepVoiceGuard) | 3 | MIT | 2025-01 | RawNet2 안티스푸핑 + FastAPI 실시간 추론 예제 |
| [AnkushRathour/AudioSpeakerVerification](https://github.com/AnkushRathour/AudioSpeakerVerification) | 1 | **GPL-3.0 ⚠** | 2024-08 | 우리 구조와 동일한 미니 예제이나 GPL — **코드 복사 금지, 구조만 참고** |

### 5.2 벡터 DB
| 저장소 | ⭐ | 라이선스 | 최근 활동 | 평가 |
| :--- | ---: | :--- | :--- | :--- |
| [pgvector/pgvector](https://github.com/pgvector/pgvector) | 22,830 | PostgreSQL License | 2026-08 | **아키텍처 핵심 그대로.** HNSW·IVFFlat, 코사인 거리 |
| [pgvector/pgvector-python](https://github.com/pgvector/pgvector-python) | 1,518 | MIT | 2026-07 | SQLAlchemy/psycopg/asyncpg 바인딩 — FastAPI 필수 동반 |
| [supabase/supabase](https://github.com/supabase/supabase) | 108,625 | Apache-2.0 | 2026-08 | 관리형 Postgres + pgvector + Auth/Storage |
| [supabase-community/nextjs-openai-doc-search](https://github.com/supabase-community/nextjs-openai-doc-search) | 1,731 | Apache-2.0 | 2026-05 | `match_documents` RPC 코사인 검색 패턴 → 화자 매칭에 이식 가능 |
| [qdrant/qdrant](https://github.com/qdrant/qdrant) | 34,280 | Apache-2.0 | 2026-08 | 대안 1. 현 규모에선 불필요한 인프라 추가 |
| [milvus-io/milvus](https://github.com/milvus-io/milvus) | 45,890 | Apache-2.0 | 2026-08 | 대안 2. 수백만 화자급에서만 고려 |

**결론:** **pgvector 유지가 정답.** 화자 수 10만 미만이면 HNSW + 코사인으로 충분하며, 사용자 계정·인증 로그·임베딩을 **단일 트랜잭션**으로 다룰 수 있다는 점이 전용 벡터 DB 대비 결정적 이점.

### 5.3 Flutter 클라이언트
| 저장소 | ⭐ | 라이선스 | 최근 활동 | 평가 |
| :--- | ---: | :--- | :--- | :--- |
| [llfbandit/record](https://github.com/llfbandit/record) | 316 | 루트 LICENSE 부재(pub.dev BSD-3) ⚠ | 2026-07 | **1순위 녹음 플러그인.** PCM 16kHz 스트림 지원이 전처리에 유리 |
| [cfug/dio](https://github.com/cfug/dio) | 12,840 | MIT | 2026-08 | `FormData`/`MultipartFile` 업로드 + 인터셉터 표준 |
| [SimformSolutionsPvtLtd/audio_waveforms](https://github.com/SimformSolutionsPvtLtd/audio_waveforms) | 347 | MIT | 2026-07 | 실시간 파형 UI. **자체 녹음기를 쓰므로 record와 이중 녹음 주의** |
| [Canardoux/flutter_sound](https://github.com/Canardoux/flutter_sound) | 941 | MPL-2.0 | 2025-11 | 대안. API가 무겁고 MPL |

### 5.4 gRPC 오디오 스트리밍 (확장 시)
| 저장소 | ⭐ | 라이선스 | 평가 |
| :--- | ---: | :--- | :--- |
| [grpc/grpc-dart](https://github.com/grpc/grpc-dart) | 892 | Apache-2.0 | Flutter gRPC 스트리밍의 유일한 공식 기반 |
| [alphacep/vosk-server](https://github.com/alphacep/vosk-server) | 1,261 | Apache-2.0 | Python 오디오 스트리밍 서버 설계(청크 프로토콜, proto) 최고 참조 |
| [nvidia-riva/python-clients](https://github.com/nvidia-riva/python-clients) | 138 | MIT | 프로덕션급 오디오 gRPC proto 설계 참조 |
| [GoogleCloudPlatform/dialogflow-flutter-grpc](https://github.com/GoogleCloudPlatform/dialogflow-flutter-grpc) | 11 | Apache-2.0 | Flutter 마이크→gRPC 스트리밍 코드가 실제로 있는 드문 예제(2021, 패턴은 유효) |

**결론:** 현 Thin Client 구조(짧은 발화 일괄 업로드)에는 **HTTP multipart(dio)로 충분**하며 gRPC는 필수가 아니다. 실시간 연속 검증으로 확장할 때 grpc-dart + vosk-server의 proto/청크 설계를 참조.

---

## 6. 종합 리스크 및 판단

1. **"완성형 참조 저장소"는 존재하지 않는다.** FastAPI + 화자검증 + pgvector를 한 번에 묶은 신뢰할 만한 저장소는 없으며, speechbrain(임베딩) + pgvector-python(검색) + supabase doc-search(SQL 패턴)를 **직접 조합하는 것이 표준 경로**다. 초기 개발 공수를 여기에 배정해야 한다.
2. **라이선스 리스크 4건:** AASIST3(CC BY-NC-ND, 상용 불가), wesep(라이선스 파일 없음), AnkushRathour 예제(GPL-3.0), record(루트 LICENSE 부재 — pub.dev BSD-3 표기 확인 필요). TEN-VAD도 Apache-2.0 변형이라 검토 대상.
3. **핵심 저장소의 커밋 정체는 대체로 수용 가능:** DeepFilterNet(2024-10), aasist(2023-06), SSL_Anti-spoofing(2023-09)은 코드가 작고 자족적이어서 포크 유지 비용이 낮다. 반면 Resemblyzer·demucs·asv-subtools는 대체재가 명확하므로 신규 채택하지 않는다.
4. **CPU 추론 가능성이 모델 선택의 실질적 제약이다.** Mamba 계열 전면 배제, SSL 기반 스푸핑 탐지의 캐스케이드 구성, DeepFilterNet 채택 모두 이 제약에서 도출됐다. GPU 예산이 확보되면 재평가한다.
