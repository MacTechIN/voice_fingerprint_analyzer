# 06_Development_Plan.md
# 개발 계획서 (Micro-process & Vertical Stackable)

앱 경량화를 목표로 하므로, **API 서버 우선 구축 후 클라이언트를 연동하는 방식**으로 스택을 쌓아 올립니다. Phase 1~5는 프로토타입(Phase A) 완성 경로이며, Phase 6~8은 연구 보고서(『음성 분리 및 화자 대조』)의 4단계 기술 로드맵 — ① 전처리·분리 프론트엔드, ② 초구면 잠재 공간 최적화, ③ 분리 왜곡 극복·AS-Norm 백엔드, ④ 보안 방어막 — 을 단계적으로 반영하는 확장 경로입니다.

## Phase 1: AI 서버 구축 및 API 설계 (Week 1-2)
*   *Micro-process:* Python(FastAPI) 기반으로 ECAPA-TDNN 모델 로드 환경 구성. 파이프라인 최전단에 Silero VAD 결합 (무음·소음 구간 제거 후 임베딩 추출).
*   *Output:* 오디오 파일을 업로드하면 임베딩 벡터(모델 기본 차원, ECAPA-TDNN 기준 192차원)를 반환하는 `POST /extract` API 완성. 유효 발화 길이 미달 시 사유 코드 반려 응답 포함.

## Phase 2: Vector DB 통합 및 인증 로직 (Week 3-4)
*   *Micro-process:* Supabase pgvector 구축 (임베딩 모델명·버전·차원 메타데이터 컬럼 포함). 두 오디오 벡터를 비교해 % 확률을 도출하는 코사인 유사도 로직 작성.
*   *Vertical Slice:* 서버 단독으로 등록(`POST /enroll`) 및 검증(`POST /verify`) API 엔드투엔드 작동 확인 (Postman 테스트).
*   *설계 원칙:* 응답 JSON은 확장 가능 스키마로 설계 (추후 `normalized_score`, `spoof_score` 등 필드 무중단 추가 대비).

## Phase 3: 경량화된 Cross-Platform App 개발 (Week 5-7)
*   *Micro-process:* Flutter 프로젝트 세팅, 마이크 제어 및 오디오 파일화 로직 구현 (16kHz/Mono 캡처, 최소 3초 발화 강제, 입력 레벨 피드백).
*   *Vertical Slice:* 앱 UI에서 버튼을 눌러 녹음 후, Phase 2에서 만든 API로 전송하여 화면에 00% 확률 결과를 출력하는 E2E 통신 완성.

## Phase 4: Web Dashboard 구축 (Week 8)
*   *Micro-process:* Next.js로 관리자 웹 구축, API 서버의 DB(Supabase)에 접근하여 통계 로그 구현.

## Phase 5: 서버 최적화 및 네트워크 지연 최소화 (Week 9-10)
*   *Micro-process:* 클라이언트 체감 속도 향상을 위해 오디오 전송 압축, FastAPI 워커(Worker) 수 조정, 네트워크 응답 지연(Latency) 개선. (필요시 gRPC 통신으로 업그레이드).

---

## Phase 6: 판정 신뢰성 강화 — 음성 향상 + AS-Norm (확장 Phase B)
*   *Micro-process 1:* DeepFilterNet 소음 억제를 VAD 다음 단계에 결합. PESQ/STOI로 향상 전후 품질 비교 검증.
*   *Micro-process 2:* 임포스터 코호트 임베딩 DB 구축(공개 데이터셋 활용, 사용자 DB와 격리) 후 AS-Norm 대칭 정규화 스코어링 구현.
*   *Micro-process 3:* 자체 평가 트라이얼 셋 구성 → EER·minDCF 측정 스크립트 작성 → 웹 대시보드에 임계값 캘리브레이션 UI 추가.
*   *Vertical Slice:* 동일 검증 요청에 대해 원시 점수와 정규화 점수를 병기 반환·로그하고, 정규화 적용 전후 EER 개선을 수치로 확인.

## Phase 7: 다중 화자 대응 — 음성 분리 / TSE (확장 Phase C)
*   *Micro-process 1:* SpeechBrain SepFormer 사전학습 모델로 2인 혼합 오디오 분리 PoC. 실시간 요건 발생 시 Mamba-TasNet 계열로 교체 검토.
*   *Micro-process 2:* 등록 임베딩을 조건으로 하는 타겟 화자 추출(TSE) 방식 실험 — 순열 문제 회피 및 파이프라인 단순화 비교.
*   *Micro-process 3:* **임베딩 왜곡 대응:** 분리망 출력 아티팩트를 첨가한 증강 데이터로 임베딩 백본 파인튜닝(탈동조화 전략). 이 시점에 백본을 단기 발화에 강한 ERes2NetV2(3D-Speaker)로 교체 평가 (교체 시 기존 사용자 재등록 마이그레이션 절차 수행).
*   *Vertical Slice:* 2인 동시 발화 녹음 → 분리 → 타겟 화자 검증 성공률을 단일 화자 대비 벤치마크.

## Phase 8: 보안 방어막 — Anti-Spoofing (확장 Phase D)
*   *Micro-process 1:* AASIST 사전학습 모델을 검증 파이프라인 초입에 병렬 배치, `spoof_score` 임계 기반 차단 로직 구현.
*   *Micro-process 2:* TTS 생성 음성으로 자체 공격 시뮬레이션 → 탐지율(EER·F1) 측정. 성능 미달 시 SSL 프론트엔드 결합형(PT-SSL-AASIST) 또는 AASIST3 업그레이드.
*   *Vertical Slice:* 합성 음성 인증 시도가 차단되고 관리자 대시보드에 스푸핑 로그·경보가 표시되는 E2E 확인.

## 리스크 및 대응 원칙
| 리스크 | 대응 |
| :--- | :--- |
| 짧은 발화(2~3초)에서 임베딩 신뢰도 저하 | 클라이언트 최소 발화 강제(FR-01) + Phase 7에서 ERes2NetV2 전환 |
| 분리 아티팩트로 인한 검증 성능 붕괴 | 아티팩트 증강 파인튜닝(탈동조화) 우선, EEND-SS 결합 학습은 장기 과제로 보류 |
| 원시 점수 고정 임계값의 높은 실패율 | Phase 6 AS-Norm을 필수 경로로 편성 |
| 모델 교체 시 기존 성문 벡터 비호환 | 벡터에 모델 버전 메타데이터 저장, 재등록 마이그레이션 플로우 사전 설계 |
