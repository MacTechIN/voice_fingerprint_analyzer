# VoiceGuard-Verification (voice_fingerprint_analyzer)

[![version](https://img.shields.io/github/v/tag/MacTechIN/voice_fingerprint_analyzer?sort=semver&label=version&color=blue)](CHANGELOG.md)
[![status](https://img.shields.io/badge/status-pre--release-orange)](CHANGELOG.md#%EB%B2%84%EC%A0%84-%EC%A0%95%EC%B1%85)
[![python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](docs/09_Operations.md)
[![flutter](https://img.shields.io/badge/flutter-3.27+-02569B?logo=flutter&logoColor=white)](app/)
[![node](https://img.shields.io/badge/node-20+-339933?logo=nodedotjs&logoColor=white)](web/)
[![license](https://img.shields.io/badge/license-Apache--2.0-D22128)](LICENSE)

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
| 딥페이크 탐지 (Phase 8) | ✅ AASIST-L (도메인 밖 오탐 문제 측정·완화) | [server/](server/) |
| 서버 최적화 (Phase 5) | ✅ 동시성 버그 수정, 처리량 +60% | [server/eval/bench.py](server/eval/bench.py) |
| Flutter 앱 (Phase 3) | ✅ 5개 OS 크로스플랫폼 클라이언트 | [app/](app/) |
| 관리자 웹 (Phase 4) | ✅ 대시보드·화자DB·오딧트레일·캘리브레이션 | [web/](web/) |

```bash
cd server && ./run.sh                                          # API: localhost:8000/docs
cd web && npm run dev                                          # 관리자: localhost:3000
cd app && flutter run --dart-define=VG_API_BASE_URL=http://localhost:8000
```

최초 설치·DB 준비·배포 전 점검은 **[09_Operations.md](docs/09_Operations.md)** 참조.
버전별 변경 내역과 검증되지 않은 항목은 **[CHANGELOG.md](CHANGELOG.md)** 참조.

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
| [09_Operations](docs/09_Operations.md) | **서비스 실행 가이드** — 설치·기동·배포 전 점검·문제 해결 |
| 음성 분리 및 화자 대조 (md/pdf) | 기반 심층 기술 연구 보고서 (수식 원본 참조용) |
| [chunked-preprocess-pattern](docs/chunked-preprocess-pattern.md) | 청크 선처리·재사용 패턴 (SEP v2/LEP v2) — **보관 자료, 향후 성능 개선용** |

## 라이선스

이 저장소의 코드와 문서는 **[Apache License 2.0](LICENSE)** 을 따른다.
특허 라이선스 조항이 포함된 허용적 라이선스이며, 채택한 핵심 스택(WeSpeaker,
3D-Speaker, SpeechBrain)과 조건이 일치한다.

**서드파티 코드가 하나 포함돼 있다.** `server/vendor/`의 AASIST 모델 소스는
[clovaai/aasist](https://github.com/clovaai/aasist)에서 수정 없이 그대로 가져온
MIT 라이선스 코드다. MIT가 요구하는 저작권·허가 문구는 [NOTICE](NOTICE)에 있으며,
이 저장소를 재배포할 때 함께 포함해야 한다.

**사전학습 모델 가중치는 이 라이선스의 적용 대상이 아니다.** 가중치는 저장소에
포함하지 않고 실행 시 내려받는다. 각 모델(WeSpeaker ResNet34-LM, silero-vad,
AASIST-L 등)은 자체 라이선스를 따르므로 배포 전 개별 확인이 필요하다. 조사 결과는
[08_OpenSource_Survey](docs/08_OpenSource_Survey.md)에 정리돼 있으며, 특히 다음
두 가지는 주의해야 한다.

- **AASIST3는 CC BY-NC-ND 4.0으로 상용 배치가 불가하다.** 채택한 것은 MIT인
  AASIST-L이며, 성능이 더 좋다는 이유로 AASIST3로 바꾸면 라이선스가 깨진다.
- **라이선스가 표기되지 않은 저장소가 있다.** `wesep`과 `record`가 그렇다.
  `record`는 pub.dev 표기(BSD-3)를 확인하고 채택했다.
