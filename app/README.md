# VoiceGuard — 크로스플랫폼 클라이언트 (Phase 3)

Flutter 단일 코드베이스로 **Android · iOS · Linux · macOS · Windows**를 지원하는
Thin Client. 녹음과 UI만 담당하고 분석은 전부 서버가 한다 — **머신러닝 라이브러리를
일절 포함하지 않는다** (01 §2, 04 §1).

## 빠른 시작

```bash
# 1. 서버를 먼저 띄운다 (../server/README.md)
cd ../server && export VG_DATABASE_URL="postgresql://..." && ./run.sh

# 2. 앱 실행
cd ../app
flutter pub get
flutter run --dart-define=VG_API_BASE_URL=http://localhost:8000
```

`VG_API_BASE_URL` 기본값은 `http://localhost:8000`이다. **Android 에뮬레이터에서는
`http://10.0.2.2:8000`**을 써야 호스트에 닿는다 (에뮬레이터의 localhost는 자기 자신이다).

## 구조

```
lib/
  api/       models.dart    서버 응답 파싱 (Tolerant Reader)
             errors.dart    반려 코드 ↔ 사용자 안내 매핑
             client.dart    dio 기반 HTTP 클라이언트
  audio/     recording_policy.dart  녹음 규격·최소 길이·레벨 판정·발화 스크립트
             recorder.dart          마이크 캡처 (인터페이스 + 실제 구현)
  state/     voice_controller.dart  녹음 → 전송 → 결과 상태 기계
  ui/        screens/voice_screen.dart  등록·인증 화면
             widgets/                   레벨 미터, 결과 시각화
```

## 데이터 플로우 (04 §5)

```
안내 문구 + 발화 스크립트
      ↓ [녹음하여 인증] 
녹음 (16kHz/Mono WAV) — 실시간 레벨 미터 + 경과 타이머
      ↓ [완료]  ← 최소 길이 미달이면 버튼이 잠긴다
길이 검사 (앱) → 미달 시 즉시 안내, 서버로 보내지 않음
      ↓
POST /api/v1/verify (Multipart) — 로딩 스피너
      ↓
결과 애니메이션 (일치도 게이지) + 임시 파일 삭제
```

## 설계 원칙

### Tolerant Reader — 서버 스키마 확장에 앱이 끌려다니지 않는다

서버는 확장 가능 스키마를 쓰며 Phase 8에서 `spoof_score` 같은 필드가 추가된다
(01 §2). 파싱은 **알 수 없는 필드를 무시**하고, 없는 필드는 null로 받는다. 필드가
늘었다고 앱이 깨지면 서버를 배포할 때마다 앱을 함께 내보내야 한다.

같은 이유로 반려 코드에 `unknown`이 있다. 서버가 새 사유를 추가해도 앱은
서버가 준 `detail` 문구를 그대로 보여주면 된다.

### 서버에 보내기 전에 앱이 먼저 막는다

짧은 녹음을 그대로 보내면 왕복 시간을 쓰고 나서야 "더 길게 말하세요"를 듣는다.
앱이 최소 길이를 검사해 즉시 안내하고, 최소 길이를 채우기 전에는 완료 버튼을 잠근다.

클라이언트 최소 길이(검증 3초 / 등록 5초)는 서버의 유효 발화 하한(1.5초)보다
넉넉하다. VAD가 무음을 걷어내면 녹음 길이보다 짧아지기 때문이다.

### 서버는 "무엇이 잘못됐는지", 앱은 "무엇을 하면 되는지"

```
서버 detail:  유효 발화가 너무 짧습니다 (0.4초). 1.5초 이상 말해주세요.
앱 actionHint: 조금 더 길게 말해주세요. 문장 하나를 천천히 읽으면 충분합니다.
```

둘을 함께 보여준다.

### 판정 신뢰도가 낮은 상태를 숨기지 않는다

서버에 AS-Norm 코호트가 없으면 원시 코사인으로 폴백하는데, 이때 판정 신뢰도가
떨어진다. 앱은 결과 화면에 경고를 띄우고, 상단 배너에도 서버 상태(정규화 꺼짐,
임시 저장소)를 드러낸다. 운영 문제를 모르고 지나치는 것이 가장 위험하다.

### 일치도는 "확률"이 아니다

`match_probability`는 판정 경계를 50%에 맞춘 구간 선형 재척도다. UI에서 **"확률"이라
쓰지 않고 "일치도"로 표기**한다. 위젯 테스트가 이를 강제한다.

### 생체정보를 기기에 남기지 않는다

전송 성공·실패와 무관하게 임시 오디오 파일을 삭제한다 (FR-02, 04 §6).
컨트롤러의 `finally` 블록에서 처리하며, 테스트가 두 경로 모두 확인한다.

## OS별 권한 (04 §3)

| 플랫폼 | 설정 |
| :--- | :--- |
| Android | `RECORD_AUDIO`, `INTERNET` (AndroidManifest.xml) |
| iOS | `NSMicrophoneUsageDescription` (Info.plist) — 없으면 앱이 즉시 종료된다 |
| macOS | `network.client`, `device.audio-input` 엔타이틀먼트 + 마이크 사용 설명 |
| Linux/Windows | 별도 선언 불필요 |

## 테스트

```bash
flutter analyze                    # 0 issues
flutter test                       # 59 passed
```

| 파일 | 대상 |
| :--- | :--- |
| `models_test.dart` | 응답 파싱, 미지 필드 내성, 정수/실수 혼용, AS-Norm 폴백 |
| `policy_test.dart` | 녹음 규격, 길이 검사, 레벨 판정, 반려 코드 매핑 |
| `client_test.dart` | HTTP 계층 모킹 — 422 반려·네트워크 오류 변환 |
| `controller_test.dart` | 상태 전이, **임시 파일 삭제**(성공·실패 양쪽) |
| `widget_test.dart` | 레벨 경고, 결과 시각화, 폴백 경고, 실패 안내 |

### 실제 서버 E2E

모킹된 계약이 아니라 **진짜 서버 응답**으로 검증한다. 서버가 스키마를 바꿨는데
앱이 모르는 상황을 여기서 잡는다.

```bash
flutter test test/e2e_live_test.dart \
  --dart-define=VG_E2E_BASE_URL=http://127.0.0.1:8000 \
  --dart-define=VG_E2E_AUDIO_DIR=/path/to/wavs
```

오디오는 `spk0_0.wav`(등록), `spk0_1.wav`(동일 화자), `spk1_0.wav`(타 화자)가 필요하다.
LibriSpeech 발화로 실측한 결과: 등록 → 동일 화자 통과 → 타 화자 거부 → 미등록
반려 → 재등록 대체가 모두 확인됐다.

`VG_E2E_BASE_URL`이 없으면 자동으로 건너뛴다 — 서버 없이도 나머지 테스트가 돈다.

## 알려진 제약

- **실기기 검증은 아직 없다.** 위젯·단위 테스트와 실제 서버 E2E는 통과했지만,
  마이크 캡처는 `FakeRecorder`로 대체했다. `record` 패키지의 실제 동작(권한 팝업,
  기기별 샘플레이트 지원)은 실기기에서 확인해야 한다.
- **사용자 ID가 하드코딩(`demo-user`)이다.** 실제 서비스에서는 로그인 세션에서 온다.
- **스푸핑 차단 UX(FR-18)가 없다.** 서버 Phase 8이 들어오면 `spoof_score` 기반
  "실제 음성 확인 실패" 안내를 추가한다. 응답 파싱은 이미 미지 필드에 내성이 있어
  서버가 먼저 배포돼도 앱이 깨지지 않는다.
- **오프라인 큐가 없다.** 네트워크가 끊기면 즉시 실패한다. Thin Client 특성상
  서버 없이는 아무것도 못 하므로 의도된 동작이다.
