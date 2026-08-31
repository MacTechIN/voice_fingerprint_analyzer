/// 실제 서버를 상대로 하는 E2E 테스트 (06 Phase 3 Vertical Slice).
///
/// 모킹된 계약이 아니라 **실제 서버 응답**으로 클라이언트를 검증한다. 서버가
/// 응답 스키마를 바꿨는데 앱이 모르고 있는 상황을 여기서 잡는다.
///
/// 서버가 떠 있을 때만 돌린다:
///
///     VG_E2E_BASE_URL=http://127.0.0.1:8000 \
///       flutter test test/e2e_live_test.dart
///
/// 오디오는 `VG_E2E_AUDIO_DIR`의 wav 파일을 쓴다. 화자별로 `spk0_0.wav`,
/// `spk0_1.wav`(동일 화자 다른 발화), `spk1_0.wav`(다른 화자)가 필요하다.
library;

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:voiceguard/api/client.dart';
import 'package:voiceguard/api/errors.dart';

const String _baseUrl = String.fromEnvironment('VG_E2E_BASE_URL');
const String _audioDir = String.fromEnvironment('VG_E2E_AUDIO_DIR');

void main() {
  if (_baseUrl.isEmpty) {
    test('E2E 건너뜀 (VG_E2E_BASE_URL 미설정)', () {}, skip: true);
    return;
  }

  final api = VoiceGuardApi(baseUrl: _baseUrl);
  final userId = 'flutter-e2e-${DateTime.now().millisecondsSinceEpoch}';

  File audio(String name) {
    final file = File('$_audioDir/$name');
    if (!file.existsSync()) {
      throw StateError('E2E 오디오 없음: ${file.path}');
    }
    return file;
  }

  test('서버가 준비되어 있다', () async {
    final health = await api.health();

    expect(health.isReady, isTrue, reason: '서버가 응답할 준비가 되어야 한다');
    expect(health.status, 'ok');
    // 서버가 어떤 백엔드로 떠 있든 앱은 동작해야 한다
    expect(health.embeddingBackend, isNotEmpty);
  });

  test('등록 → 동일 화자 검증 → 타 화자 거부', () async {
    // 1. 등록
    final enrolled = await api.enroll(userId: userId, audio: audio('spk0_0.wav'));
    expect(enrolled.userId, userId);
    expect(enrolled.enrollmentId, greaterThan(0));
    expect(enrolled.isReEnrollment, isFalse);
    expect(enrolled.audio.speechDurationSec, greaterThan(1.5));

    // 2. 같은 화자의 다른 발화 — 통과해야 한다
    final same = await api.verify(userId: userId, audio: audio('spk0_1.wav'));
    expect(same.isVerified, isTrue,
        reason: '동일 화자의 다른 발화는 본인으로 판정되어야 한다');
    expect(same.matchProbability, greaterThan(50),
        reason: '판정 경계가 50%이므로 통과했다면 50%를 넘어야 한다');

    // 3. 다른 화자 — 거부해야 한다
    final other = await api.verify(userId: userId, audio: audio('spk1_0.wav'));
    expect(other.isVerified, isFalse, reason: '다른 화자는 거부되어야 한다');
    expect(other.rawCosine, lessThan(same.rawCosine));
  });

  test('미등록 사용자는 not_enrolled로 반려된다', () async {
    await expectLater(
      api.verify(userId: 'never-enrolled-xyz', audio: audio('spk0_0.wav')),
      throwsA(isA<ApiException>()
          .having((e) => e.code, 'code', RejectionCode.notEnrolled)
          .having((e) => e.statusCode, 'statusCode', 422)),
    );
  });

  test('재등록은 기존 성문을 대체한다', () async {
    final again = await api.enroll(userId: userId, audio: audio('spk0_1.wav'));

    expect(again.isReEnrollment, isTrue);
    expect(again.replaced, greaterThan(0));
  });
}
