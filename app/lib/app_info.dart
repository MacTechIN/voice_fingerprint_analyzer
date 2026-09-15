/// 앱 식별 정보.
///
/// 버전은 pubspec의 값과 같아야 한다. 두 곳에 적는 대신 한쪽을 읽어오는 방법은
/// 패키지 의존성이 필요해, 상수로 두고 `test/license_test.dart`가 어긋남을
/// 잡도록 했다.
library;

const String appName = 'VoiceGuard';
const String appVersion = '0.9.0';
