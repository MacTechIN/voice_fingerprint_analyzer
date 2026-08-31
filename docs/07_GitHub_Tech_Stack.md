# 07_GitHub_Tech_Stack.md
# 클라이언트 경량화 및 서버 분석 아키텍처 관련 깃허브 조사 결과

앱을 가볍게 유지하고 외부 서버에서 분석을 처리하기 위한 오픈소스 채택안입니다.
아래 내용은 **GitHub API 실측 조사(2026-08-31)** 결과를 반영한 것으로, 저장소별 상세 비교표와 탈락 사유는 [08_OpenSource_Survey.md](08_OpenSource_Survey.md)를 참조하십시오.

## 1. 서버 코어 (프로토타입 Phase A)

1. **SpeechBrain — 화자 인증 프로토타입 최속 경로**
   * **저장소:** `speechbrain/speechbrain` (11.8k★, Apache-2.0, v1.1.1 / 2026-08)
   * **채택 사유:** `SpeakerRecognition.from_hparams("speechbrain/spkrec-ecapa-voxceleb")` 한 줄로 등록·검증 데모가 동작하여 Phase 1 착수 속도가 압도적입니다. 영어권 문서·커뮤니티가 가장 좋고 유지보수 신뢰도도 최상급입니다.
   * **한계 (중요):** **AS-Norm 스코어 정규화와 서빙 계층이 없습니다.** 따라서 Phase 6 시점에 WeSpeaker로 코어를 전환하는 것을 전제로 채택합니다. 또한 recipes 44종 전수 확인 결과 **anti-spoofing 레시피가 존재하지 않으므로** Phase 8 용도로는 사용할 수 없습니다.

2. **Silero VAD — 전처리 최전단**
   * **저장소:** `snakers4/silero-vad` (10.1k★, MIT, 2026-08 활발)
   * **채택 사유:** 30ms 청크당 약 1ms(CPU 1스레드)로 실시간 요건을 충족하며, ONNX 모델이 동봉되어 onnxruntime만으로 의존성 최소 배포가 가능합니다. 무음·소음 구간이 임베딩을 오염시키는 것을 차단하는 파이프라인 1단계 필수 모듈입니다.
   * **대안 검토:** `TEN-framework/ten-vad`가 지연·정확도 우위를 주장하나 **Apache-2.0에 추가 조건이 붙은 변형 라이선스**라 상용 전 법무 검토가 필요합니다.

## 2. 서버 본구축 (Phase B 이후)

3. **WeSpeaker — 코어 엔진 (임베딩 + 정규화 + 서빙)**
   * **저장소:** `wenet-e2e/wespeaker` (1.4k★, Apache-2.0, 2026-07)
   * **채택 사유:** 조사 대상 중 **유일하게 "훈련 → AS-Norm/PLDA/캘리브레이션 → ONNX export → C++ onnxruntime 런타임 → Triton GPU 서버"가 한 저장소에서 완결**됩니다. `wespeaker/bin/score_norm.py`, `score_calibration.py`, `runtime/server/x86_gpu` 존재를 직접 확인했습니다. ECAPA-TDNN, CAM++, ERes2Net, ReDimNet, Whisper-PMFA 등 최신 아키텍처를 지원합니다.
   * **업계 표준 방증:** pyannote-audio 4.x조차 화자 임베딩을 WeSpeaker ResNet34 포팅으로 가져다 씁니다.

4. **3D-Speaker — 단기 발화 강건 임베딩 공급원 (Phase C)**
   * **저장소:** `modelscope/3D-Speaker` (3.1k★, Apache-2.0, 2025-12)
   * **채택 사유:** ERes2NetV2·CAM++ 원저작(Alibaba) 공식 구현으로, 음성 분리 직후 2~3초로 분절된 짧은 오디오에서의 임베딩 품질이 독보적입니다(VoxCeleb1-O EER 0.61%). ONNX export 스크립트가 있어 WeSpeaker 서빙 스택에 얹기 쉽습니다.
   * **유의사항:** **AS-Norm 스크립트가 없으므로** 정규화는 WeSpeaker 구현을 이식해야 합니다.

## 3. 전처리 프론트엔드 (Phase B~C)

5. **DeepFilterNet — 실시간 음성 향상**
   * **저장소:** `Rikorose/DeepFilterNet` (4.6k★, MIT + Apache-2.0 듀얼)
   * **채택 사유:** CPU RTF ≈0.04로 실시간 인증 시나리오에 부합하는 유일한 선택지이며 ONNX export 경로가 있습니다. 무엇보다 **판별형(discriminative) 모델이라 생성형(resemble-enhance 등) 대비 성문 왜곡 위험이 낮다는 점**이 화자 인증 전처리에서 결정적입니다.
   * **유의사항:** 2024-10 이후 커밋이 정체되어 있으나, 코드가 자족적이고 성숙 단계로 판단되어 포크 유지 비용은 낮습니다.

6. **ClearerVoice-Studio — 음성 분리 (Phase C)**
   * **저장소:** `modelscope/ClearerVoice-Studio` (4.5k★, Apache-2.0, 2025-08)
   * **채택 사유:** MossFormer2 계열 분리(SS)·향상(SE)·TSE를 한 툴킷에서 제공해 파이프라인 통합 비용이 최소이고, 16kHz 사전학습 모델이 본 시스템 샘플레이트와 일치합니다. SepFormer 대비 최신 SOTA입니다.
   * **대안:** 프레임워크를 SpeechBrain으로 통일할 경우 SepFormer가 역전 가능합니다.
   * **배제:** `facebookresearch/demucs`는 **저장소가 아카이브**되었고 음악 소스 분리 특화라 화자 분리 목적에 부적합합니다. Mamba 계열(SPMamba/SEMamba)은 `mamba-ssm`의 **CUDA 의존으로 CPU 추론이 불가**하여 현시점 채택 대상이 아닙니다.

7. **wesep — 타겟 화자 추출(TSE) *(조건부)***
   * **저장소:** `wenet-e2e/wesep` (311★, **라이선스 파일 없음 ⚠**, 2025-10)
   * **채택 사유:** TSE 전용 툴킷 중 유일하게 화자 임베딩(WeSpeaker)과 추출 모델의 joint training을 지원하여, 등록 임베딩을 단서로 쓰는 우리 구조와 개념적으로 정확히 일치합니다.
   * **선행 조건:** **라이선스 확인 전 채택 불가.** 커뮤니티 규모도 작아 자체 유지 역량이 필요합니다.

## 4. 보안 방어막 (Phase D)

8. **AASIST / SSL_Anti-spoofing — 딥페이크 탐지 2단 캐스케이드**
   * **저장소:** `clovaai/aasist` (295★, MIT) + `TakHemlata/SSL_Anti-spoofing` (179★, MIT)
   * **채택 사유:** 1차 스크리닝에 AASIST-L(85K 파라미터, ASVspoof2019 LA EER 0.99%, CPU 실시간), 2차 정밀 판정에 XLSR+AASIST(ASVspoof2021 LA EER 0.82%, DF 2.85%)를 배치하여 정확도와 지연 예산을 동시에 만족시킵니다. 두 저장소 모두 MIT로 상용 안전합니다.
   * **통합 유의사항:** SSL_Anti-spoofing은 고정 커밋 fairseq / torch 1.8에 의존하므로, 추론 전용인 본 프로젝트에서는 프론트엔드를 HuggingFace `wav2vec2-xls-r-300m`으로 교체하는 경량 포팅을 권장합니다.
   * **라이선스 배제:** `lab260ru/AASIST3`는 **CC BY-NC-ND 4.0**으로 상용 이용과 파생물 제작이 모두 금지되어 채택할 수 없습니다.
   * **검증 인프라:** `MuSAELab/AUDDT`(33개 딥페이크 데이터셋 통합 평가), `asvspoof-challenge/2021` eval-package(t-DCF/EER 표준 산출).

## 5. 데이터 계층 및 서빙

9. **pgvector — 성문 벡터 저장소**
   * **저장소:** `pgvector/pgvector` (22.8k★, PostgreSQL License) + `pgvector/pgvector-python` (1.5k★, MIT)
   * **채택 사유:** 화자 수 10만 미만 규모에서 HNSW 인덱스 + 코사인 거리로 충분하며, **사용자 계정·인증 로그·임베딩을 단일 트랜잭션으로 다룰 수 있다는 점**이 전용 벡터 DB(Qdrant 34.3k★ / Milvus 45.9k★) 대비 결정적 이점입니다. 수백만 화자 규모 도달 시에만 재검토합니다.
   * **검색 패턴 참조:** `supabase-community/nextjs-openai-doc-search`의 `match_documents` RPC 코사인 검색 패턴을 화자 매칭에 이식합니다.

10. **NVIDIA Triton Inference Server — 대규모 서빙 (트래픽 증가 시)**
    * **저장소:** `triton-inference-server/server` (11.0k★, BSD-3, 2026-08)
    * **채택 사유:** 동적 배칭·멀티프레임워크(ONNX/PyTorch/TensorRT) 지원으로 대규모 동시 요청 처리에 가장 검증된 아키텍처입니다.
    * **도입 시점:** **초기 MVP에는 과합니다.** ONNX Runtime 전환을 먼저 거친 뒤 트래픽 지표를 보고 도입합니다. 중간 단계로 `k2-fsa/sherpa-onnx`(14.5k★, Apache-2.0, 매우 활발)를 두면 WeSpeaker/3D-Speaker ONNX 모델을 다언어·websocket으로 바로 서빙할 수 있습니다.

## 6. 클라이언트 (Flutter Thin-client)

11. **record + dio + audio_waveforms**
    * **저장소:** `llfbandit/record` (316★) + `cfug/dio` (12.8k★, MIT) + `SimformSolutionsPvtLtd/audio_waveforms` (347★, MIT)
    * **채택 사유:** record는 PCM 16kHz 스트림을 지원해 서버 전처리 규격과 직결되고, dio의 `FormData`/`MultipartFile`은 녹음 파일 업로드와 인터셉터 처리의 표준입니다.
    * **유의사항 2건:** ① record는 **루트에 LICENSE 파일이 없어** GitHub API가 라이선스를 인식하지 못하므로, 채택 전 pub.dev의 패키지별 BSD-3-Clause 표기를 확인해야 합니다. ② audio_waveforms는 **자체 녹음기를 내장하므로 record와 이중 녹음이 발생하지 않도록** 역할을 분리해야 합니다.

12. **gRPC 스트리밍 — 확장 시에만**
    * **저장소:** `grpc/grpc-dart` (892★, Apache-2.0) + `alphacep/vosk-server` (1.3k★, Apache-2.0, proto·청크 설계 참조)
    * **판단:** 현 Thin Client 구조(짧은 발화 일괄 업로드)에는 **HTTP multipart(dio)로 충분하며 gRPC는 필수가 아닙니다.** 실시간 연속 검증으로 확장할 때 도입합니다.

## 7. 참조 구현 관련 정정 사항

* **`yeyupiaoling/VoiceprintRecognition-Pytorch` (1.3k★, Apache-2.0):** 저장소 구조를 직접 확인한 결과 **서버 코드가 포함되어 있지 않습니다** (README에서 API 구성을 DIY 가이드로만 언급). 따라서 "RESTful 서버 배포 설계"로 인용할 수 없으며, 학습·추론 파이프라인 및 등록-인식 흐름 **참조용**으로만 사용합니다.
* **완성형 참조 저장소는 존재하지 않습니다.** FastAPI + 화자검증 + pgvector를 한 번에 묶은 신뢰할 만한 저장소는 조사 결과 없었습니다(`AnkushRathour/AudioSpeakerVerification`는 구조가 동일하나 ★1 + **GPL-3.0**이라 코드 복사 불가, 구조 참고만 가능). SpeechBrain(임베딩) + pgvector-python(검색) + Supabase RPC 패턴을 직접 조합하는 것이 표준 경로이며, 이 공수를 Phase 1~2에 별도 배정해야 합니다.
* **비권장 판정:** `Resemblyzer`(2023-10 개발 중단, GE2E 구세대 정확도), `clovaai/voxceleb_trainer`(연구용 학습 프레임워크), `Snowdar/asv-subtools`(Kaldi 종속·활동 저하), `PaddlePaddle/PaddleSpeech`(Paddle 생태계 종속 — 단 `paddlespeech server`의 화자 임베딩 HTTP 서비스는 설계 참조 가치 있음).

## 8. 스택-단계 매핑 요약

| 파이프라인 단계 | 채택 스택 | 라이선스 | 도입 시점 |
| :--- | :--- | :--- | :--- |
| VAD | Silero VAD | MIT | Phase 1 |
| 임베딩 (프로토타입) | SpeechBrain ECAPA-TDNN | Apache-2.0 | Phase 1 |
| 벡터 DB | pgvector + pgvector-python | PostgreSQL / MIT | Phase 2 |
| 클라이언트 | record + dio + audio_waveforms | BSD-3 / MIT / MIT | Phase 3 |
| 임베딩 (본구축) + AS-Norm | **WeSpeaker** | Apache-2.0 | Phase 6 |
| 음성 향상 | DeepFilterNet v3 | MIT/Apache-2.0 | Phase 6 |
| 음성 분리 | ClearerVoice-Studio (MossFormer2-SS) | Apache-2.0 | Phase 7 |
| 임베딩 모델 교체 | 3D-Speaker ERes2NetV2 | Apache-2.0 | Phase 7 |
| TSE | wesep *(라이선스 확인 후)* | **미확인 ⚠** | Phase 7 |
| Anti-Spoofing 1차/2차 | AASIST-L / SSL_Anti-spoofing | MIT | Phase 8 |
| ONNX 배포 계층 | sherpa-onnx | Apache-2.0 | Phase 5~ |
| 대규모 서빙 | Triton Inference Server | BSD-3 | 트래픽 증가 시 |
