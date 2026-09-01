# VoiceGuard-Verification (voice_fingerprint_analyzer)

단일 채널 오디오에서 화자의 성문(Voiceprint)을 추출·대조하는 **서버 분석형 화자 인증(Speaker Verification) 시스템** 프로토타입.

- **Thin Client:** Flutter 크로스플랫폼 앱(Windows/Android/Linux/iOS/MacOS)은 녹음과 API 통신만 담당
- **Fat Server:** Python(FastAPI) AI 추론 서버가 VAD → 임베딩 추출(ECAPA-TDNN → ERes2NetV2) → 코사인 유사도 + AS-Norm 정규화 판정 수행
- **확장 로드맵:** 음성 향상(DeepFilterNet) → 음성 분리/타겟 화자 추출(TSE) → 딥페이크 탐지(AASIST)

## 구현 현황

| 컴포넌트 | 상태 | 위치 |
| :--- | :--- | :--- |
| AI 서버 (Phase 1) | ✅ VAD + ECAPA-TDNN 임베딩 추출 API | [server/](server/) |
| 벡터 DB · 등록/검증 (Phase 2) | ✅ pgvector 저장 + 1:1 코사인 검증 | [server/](server/) |
| AS-Norm · 캘리브레이션 (Phase 6) | ✅ EER 1.25% 실측, WeSpeaker ONNX 코어 | [server/eval/](server/eval/) |
| 다중 화자 분리 (Phase 7) | ✅ 혼합 EER 15.6% → 6.3% (분리+타겟 선택) | [server/](server/) |
| 딥페이크 탐지 (Phase 8) | ✅ AASIST-L, ASVspoof EER 0.70% | [server/](server/) |
| 서버 최적화 (Phase 5) | ✅ 동시성 버그 수정, 처리량 +60% | [server/eval/bench.py](server/eval/bench.py) |
| Flutter 앱 (Phase 3) | ✅ 5개 OS 크로스플랫폼 클라이언트 | [app/](app/) |
| 관리자 웹 (Phase 4) | ✅ 대시보드·화자DB·오딧트레일·캘리브레이션 | [web/](web/) |

```bash
cd server && ./run.sh                                          # API: localhost:8000/docs
cd app && flutter run --dart-define=VG_API_BASE_URL=http://localhost:8000
cd web && npm run dev                                          # 관리자: localhost:3000
```

## 문서 (docs/)

| 문서 | 내용 |
| :--- | :--- |
| [01_Project_Definition](docs/01_Project_Definition.md) | 프로젝트 정의, 3대 핵심 모듈, 단계별 확장 범위 |
| [02_Technical_Specification](docs/02_Techincal_Sepcification.md) | 통신 규격, 신호 모델·SI-SNR·ArcFace·AS-Norm 수식 포함 기술 사양 |
| [03_WebApp_Definition](docs/03_WebApp_Definition.md) | 관리자 대시보드 (EER/FAR/FRR 모니터링, 임계값 캘리브레이션) |
| [04_NativeApp_Definition](docs/04_NativeApp_Definition.md) | Flutter 네이티브앱 정의, 녹음 품질 가이드 |
| [05_Functional_Requirements](docs/05_Functional_Requirements.md) | 기능 요구사항 FR-01~FR-18 (Phase A~D) |
| [06_Development_Plan](docs/06_Development_Plan.md) | Phase 1~8 개발 계획 및 리스크 대응 |
| [07_GitHub_Tech_Stack](docs/07_GitHub_Tech_Stack.md) | 오픈소스 스택 선정 및 단계 매핑 |
| [08_OpenSource_Survey](docs/08_OpenSource_Survey.md) | **오픈소스 전수 조사 (GitHub API 실측)** — 채택 스택·라이선스 리스크 |
| [09_BEF2.0_Integration](docs/09_BEF2.0_Integration.md) | BEF2.0 청크 처리 방식 도입 검토 체크리스트 (**미확정**) |
| 음성 분리 및 화자 대조 (md/pdf) | 기반 심층 기술 연구 보고서 (수식 원본 참조용) |
| [chunked-preprocess-pattern](docs/chunked-preprocess-pattern.md) | 청크 선처리·재사용 패턴 (SEP v2/LEP v2) — **보관 자료, 향후 성능 개선용** |
