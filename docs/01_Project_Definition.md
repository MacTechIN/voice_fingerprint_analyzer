# 01_Project_Definition.md
# 프로젝트 정의서: VoiceGuard-Verification 프로토타입 (서버 분석형)

## 1. 프로젝트 개요
모바일/데스크톱 클라이언트는 사용자 인터페이스(UI)와 음성 녹음 기능만을 수행하며, 추출된 오디오 데이터를 외부 AI 서버로 전송하여 사람의 목소리 성문(Voiceprint)을 분석 및 식별하는 'API 기반 화자 인증(Speaker Verification)' 시스템 프로토타입 개발.

본 프로젝트는 이른바 '칵테일 파티 문제(Cocktail Party Problem)' — 배경 소음과 타인의 간섭 음성이 섞인 단일 채널(Single-channel) 오디오에서 타겟 화자의 발화만을 추출해 신원을 확인하는 문제 — 를 실용 수준에서 해결하는 것을 장기 지향점으로 하며, 프로토타입 단계에서는 서버 집중형 화자 인증 파이프라인을 우선 구축한다.

## 2. 프로젝트 목표
*   **앱 경량화 (Thin Client):** 클라이언트 기기의 리소스(CPU, 메모리, 배터리) 소모를 최소화하기 위해 앱 내 딥러닝 추론 배제.
*   **서버 집중형 분석 (Fat Server):** 고성능 외부 서버(GCP, AWS 등)에서 성문 추출 및 벡터 변환(Embedding)을 일괄 처리.
*   **API 기반 검증 (확장 지향 설계):** 클라이언트가 음성을 서버로 전송하면, 서버가 기존 목소리 지문 DB와 분석하여 동일인 일치 확률(%)을 클라이언트에게 JSON 형태로 실시간 반환. 단, 이 검증 API는 기능 보강·추가가 예정된 확장 포인트이므로 처음부터 여지(Room)를 남겨 설계한다:
    *   **버전드 엔드포인트:** `/api/v1/...` 경로 버저닝으로 하위호환을 유지하며 기능 추가.
    *   **확장 가능한 응답 스키마:** 클라이언트는 알 수 없는 JSON 필드를 무시하도록 구현(Tolerant Reader) — 이후 `normalized_score`(AS-Norm), `spoof_score`(딥페이크 탐지), `speech_duration`(유효 발화 길이), 다중 화자 분리 결과 등 필드를 무중단 추가.
    *   **보강 후보 기능:** 1:N 화자 식별(Identification), 다중 등록 발화 기반 평균 성문, 검증 이력 기반 적응형 임계값, 스트리밍(gRPC) 실시간 검증 등.
*   **판정 신뢰성 확보:** 원시 코사인 유사도에 고정 임계값을 적용하는 단순 판정(Hard thresholding)을 지양하고, AS-Norm 기반 점수 정규화를 통해 오수락률(FAR)/오거부률(FRR)을 통제 (EER·minDCF 지표 관리).
*   **보안 무결성 (확장 목표):** TTS/보이스 컨버전 기반 딥페이크 음성으로 인증을 우회하는 논리적 접근(Logical Access) 공격에 대비한 오디오 딥페이크 탐지(Anti-Spoofing) 모듈의 병렬 결합.

## 3. 핵심 아키텍처 방향성
*   **클라이언트 (Cross-Platform):** Flutter 기반으로 Windows, Android, Linux, iOS, MacOS 지원. 오디오 녹음 및 RESTful/gRPC API 통신에 집중.
*   **서버 (AI Inference & DB):** Python(FastAPI) 기반의 AI 추론 서버 및 벡터 데이터베이스. 멀티스레딩 및 비동기 처리로 다중 클라이언트 요청 수용.

## 4. 서버 분석 파이프라인 (3대 핵심 모듈)
연구 보고서(『음성 분리 및 화자 대조』)의 통합 파이프라인 설계에 따라, 서버 측 분석은 다음 세 모듈로 구성한다.

1.  **전처리 및 음성 분리 (Front-end):** VAD(Silero VAD 등)로 비음성 구간을 제거하고, 음성 향상(DeepFilterNet)으로 배경 소음을 억제. 다중 화자가 겹친(Overlapped) 오디오는 음성 분리(Speech Separation) 모델로 개별 화자 발화를 복원. *(프로토타입: VAD만 적용 → 확장 단계: 향상·분리 모듈 순차 도입)*
2.  **화자 임베딩 (Speaker Embedding):** 가변 길이 발화 파형을 고정 차원의 조밀한 벡터(성문 패턴)로 압축. 동일 화자 벡터는 밀집(Intra-class compactness)하고 타 화자 군집과는 멀어지도록(Inter-class discrepancy) 학습된 백본(ECAPA-TDNN → ERes2NetV2) 사용.
3.  **대조 비교 (Speaker Verification Back-end):** 등록 임베딩과 검증 임베딩 간 코사인 유사도 산출 후, 임포스터 코호트 기반 AS-Norm 점수 정규화를 거쳐 동일인 여부를 최종 판별.

## 5. 프로토타입 범위 및 단계적 확장
| 단계 | 범위 | 비고 |
| :--- | :--- | :--- |
| **Phase A (프로토타입)** | VAD + 화자 임베딩 + 코사인 유사도 1:1 검증 | 단일 화자 녹음 전제 |
| **Phase B (품질 강화)** | 음성 향상(DeepFilterNet) + AS-Norm 점수 정규화 | 실환경 소음·점수 편차 대응 |
| **Phase C (다중 화자)** | 음성 분리(Speech Separation) 및 타겟 화자 추출(TSE) | 겹친 발화·회의 오디오 대응 |
| **Phase D (보안)** | 딥페이크 탐지(AASIST 계열) 전면 배치 | 합성 음성 우회 공격 차단 |
