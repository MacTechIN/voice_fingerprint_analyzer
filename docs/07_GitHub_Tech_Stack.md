# 07_GitHub_Tech_Stack.md
# 클라이언트 경량화 및 서버 분석 아키텍처 관련 깃허브 검색 결과

앱을 가볍게 유지하고 외부 서버에서 분석을 처리하기 위한 최신 아키텍처 오픈소스 사례입니다.

## 1. 서버 코어 (프로토타입 Phase A)

1. **FastAPI 기반 고성능 오디오 서빙 서버 (AI Backend)**
   * **검색 결과:** `yeyupiaoling/VoiceprintRecognition-PyTorch`
   * **증거 및 사유:** PyTorch 모델을 사용하여 화자 인식을 수행하며, 텍스트와 무관하게(Text-independent) 화자 특징을 추출합니다. 특히 RESTful API 형태로 서버 측 배포가 가능하도록 설계되어 있어, 본 프로젝트의 "서버 집중형 분석" 설계에 완벽히 부합합니다.

2. **SpeechBrain 화자 인증 API 파이프라인 (Hugging Face)**
   * **검색 결과:** `speechbrain/speechbrain` (Templates/Speaker_Verification)
   * **증거 및 사유:** 외부 서버에서 작동시킬 때 가장 빠르고 정확하게 사전 학습된 화자 인증(ECAPA-TDNN) 파이프라인을 API로 감쌀 수 있는 공식 레퍼런스·튜토리얼 포함. 동일 프레임워크에 SepFormer(음성 분리)까지 묶여 있어 Phase 7 분리 PoC와 결합 학습 프로토타이핑에도 그대로 재사용 가능.

3. **Silero VAD (전처리 최전단)**
   * **검색 결과:** `snakers4/silero-vad`
   * **증거 및 사유:** 엄격한 메모리·지연 제약에서도 견고한 경량 VAD. 무음·소음 구간이 임베딩을 오염시키는 것을 차단하는 파이프라인 1단계 필수 모듈. ONNX 지원으로 FastAPI 서버에 의존성 부담 없이 통합됩니다.

4. **NVIDIA Triton Inference Server 연동 사례 (서버 고도화)**
   * **검색 결과:** `triton-inference-server/server`
   * **증거 및 사유:** 앱 클라이언트가 가벼워지면 서버에 부하가 집중됩니다. 수만 명의 동시 요청을 처리하기 위해 딥러닝 모델(PyTorch/ONNX)을 외부 서버(GCP 등)에 배포할 때 가장 검증된 대규모 추론 아키텍처입니다.

## 2. 클라이언트 (Flutter Thin-client)

5. **Flutter 클라이언트 - API 간 멀티파트 오디오 통신**
   * **검색 결과:** `cfug/dio` (Flutter HTTP 통신 표준) + `rrousselGit/riverpod`
   * **증거 및 사유:** 클라이언트 기기 내 딥러닝 엔진을 없앤 대신, 끊김 없는 네트워크 통신이 핵심이 되었습니다. Dio 패키지의 `FormData`를 이용해 녹음된 오디오를 서버로 빠르고 안정적으로 청크 전송하는 패턴이 현대 Flutter Thin-client 앱의 표준입니다.

## 3. 확장 단계 (Phase B~D) 오픈소스 스택

6. **WeSpeaker — 스코어링·정규화 백엔드 및 오케스트레이션 (Phase B)**
   * **검색 결과:** `wenet-e2e/wespeaker`
   * **증거 및 사유:** ResNet·ECAPA-TDNN 지원, Kaldi 형식 호환에 더해 **AS-Norm·PLDA 스코어 정규화가 기본 내장** — Phase 6의 코호트 기반 대칭 정규화를 직접 구현하지 않고 검증된 구현체를 활용할 수 있습니다. 전체 파이프라인의 오케스트레이션 도구로도 권장됩니다.

7. **DeepFilterNet — 실시간 음성 향상 (Phase B)**
   * **검색 결과:** `Rikorose/DeepFilterNet`
   * **증거 및 사유:** PyTorch/ONNX 기반, 8~48kHz 지원, CPU 실시간 작동이 가능한 경량 저지연 소음 억제 프레임워크. 임베딩 추출 전 배경 소음 오염을 1차 차단하는 전처리 표준 구현체입니다.

8. **3D-Speaker — 단기 발화 강건 임베딩 백본 (Phase C)**
   * **검색 결과:** `modelscope/3D-Speaker` (Alibaba Damo Academy)
   * **증거 및 사유:** ERes2NetV2·CAM++ 등 최신 임베딩 모델 제공. 음성 분리 직후 2~3초로 분절된 짧은 오디오에서 왜곡 없는 고품질 임베딩 추출 성능이 독보적(VoxCeleb1-O EER 0.61%)이어서, Phase 7의 백본 교체 후보 1순위입니다. 온라인(실시간) 특징 추출로 디스크 I/O 병목도 제거합니다.

9. **AASIST — 오디오 딥페이크 탐지 (Phase D)**
   * **검색 결과:** `clovaai/aasist`
   * **증거 및 사유:** 스펙트로-템포럴 그래프 어텐션 기반 Anti-Spoofing SOTA 계열의 공식 PyTorch 구현. TTS/보이스 컨버전 합성 음성의 논리적 접근(LA) 공격을 검증 파이프라인 초입에서 차단하는 Phase 8 보안 모듈의 베이스라인입니다. (고도화 시 SSL 프론트엔드 결합형 PT-SSL-AASIST / AASIST3 계열로 업그레이드.)

## 4. 스택-단계 매핑 요약

| 파이프라인 단계 | 채택 스택 | 도입 시점 |
| :--- | :--- | :--- |
| VAD | Silero VAD | Phase 1 |
| 임베딩 추출 | SpeechBrain ECAPA-TDNN → 3D-Speaker ERes2NetV2 | Phase 1 → Phase 7 |
| 스코어 정규화 | WeSpeaker (AS-Norm) | Phase 6 |
| 음성 향상 | DeepFilterNet | Phase 6 |
| 음성 분리 / TSE | SpeechBrain SepFormer (PoC) | Phase 7 |
| Anti-Spoofing | AASIST | Phase 8 |
| 대규모 서빙 | Triton Inference Server | 트래픽 증가 시 |
