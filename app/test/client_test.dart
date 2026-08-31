/// API 클라이언트 테스트 — HTTP 계층을 모킹해 오류 변환을 검증한다.
library;

import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http_mock_adapter/http_mock_adapter.dart';
import 'package:voiceguard/api/client.dart';
import 'package:voiceguard/api/errors.dart';

void main() {
  late Dio dio;
  late DioAdapter adapter;
  late VoiceGuardApi api;
  late File audio;

  setUp(() {
    dio = Dio(BaseOptions(
      baseUrl: 'http://test.local',
      validateStatus: (status) => status != null && status < 500,
    ));
    adapter = DioAdapter(dio: dio);
    api = VoiceGuardApi(dio: dio);

    // Multipart 업로드에 실제 파일이 필요하다
    audio = File('${Directory.systemTemp.path}/vg_test_${DateTime.now().microsecondsSinceEpoch}.wav')
      ..writeAsBytesSync(List<int>.filled(64, 0));
  });

  tearDown(() {
    if (audio.existsSync()) audio.deleteSync();
  });

  group('verify', () {
    test('성공 응답을 파싱한다', () async {
      adapter.onPost(
        '/api/v1/verify',
        (server) => server.reply(200, {
          'status': 'success',
          'user_id': 'alice',
          'is_verified': true,
          'match_probability': 92.5,
          'raw_cosine': 0.77,
          'normalized_score': 9.14,
          'scoring_method': 'as_norm',
          'threshold': 2.9673,
          'compared_enrollments': 1,
          'audio': {'speech_duration_sec': 3.1},
          'elapsed_ms': 116,
        }),
        data: Matchers.any,
      );

      final result = await api.verify(userId: 'alice', audio: audio);

      expect(result.isVerified, isTrue);
      expect(result.matchProbability, 92.5);
      expect(result.scoringMethod, 'as_norm');
    });

    test('422 반려를 사유 코드가 담긴 예외로 바꾼다', () async {
      adapter.onPost(
        '/api/v1/verify',
        (server) => server.reply(422, {
          'status': 'error',
          'code': 'speech_too_short',
          'detail': '유효 발화가 너무 짧습니다 (0.4초). 1.5초 이상 말해주세요.',
        }),
        data: Matchers.any,
      );

      await expectLater(
        api.verify(userId: 'alice', audio: audio),
        throwsA(isA<ApiException>()
            .having((e) => e.code, 'code', RejectionCode.speechTooShort)
            .having((e) => e.statusCode, 'statusCode', 422)
            .having((e) => e.message, 'message', contains('0.4초'))
            .having((e) => e.isNetworkError, 'isNetworkError', isFalse)),
      );
    });

    test('미등록 사용자 반려를 구분한다', () async {
      adapter.onPost(
        '/api/v1/verify',
        (server) => server.reply(422, {
          'status': 'error',
          'code': 'not_enrolled',
          'detail': '등록된 성문이 없습니다.',
        }),
        data: Matchers.any,
      );

      await expectLater(
        api.verify(userId: 'ghost', audio: audio),
        throwsA(isA<ApiException>()
            .having((e) => e.code, 'code', RejectionCode.notEnrolled)
            .having((e) => e.isRetryableByReRecording, 'retryable', isFalse)),
      );
    });

    test('앱이 모르는 새 사유 코드도 서버 문구를 살려 전달한다', () async {
      // 서버가 앞으로 추가할 사유를 가정한다. 실재하는 코드를 쓰면 앱이 그 코드를
      // 알게 된 순간 이 테스트가 의미를 잃는다 (실제로 spoof_detected가 그랬다).
      adapter.onPost(
        '/api/v1/verify',
        (server) => server.reply(422, {
          'status': 'error',
          'code': 'replay_attack_detected_v3',
          'detail': '알 수 없는 이유로 거부되었습니다.',
        }),
        data: Matchers.any,
      );

      await expectLater(
        api.verify(userId: 'alice', audio: audio),
        throwsA(isA<ApiException>()
            .having((e) => e.code, 'code', RejectionCode.unknown)
            .having((e) => e.message, 'message', '알 수 없는 이유로 거부되었습니다.')),
      );
    });

    test('합성 음성 차단을 사유 코드로 구분한다', () async {
      adapter.onPost(
        '/api/v1/verify',
        (server) => server.reply(422, {
          'status': 'error',
          'code': 'spoof_detected',
          'detail': '실제 음성으로 확인되지 않았습니다. 직접 말씀해주세요.',
        }),
        data: Matchers.any,
      );

      await expectLater(
        api.verify(userId: 'alice', audio: audio),
        throwsA(isA<ApiException>()
            .having((e) => e.code, 'code', RejectionCode.spoofDetected)
            .having((e) => e.actionHint, 'actionHint', contains('직접'))),
      );
    });

    test('네트워크 오류는 별도로 표시한다', () async {
      adapter.onPost(
        '/api/v1/verify',
        (server) => server.throws(
          0,
          DioException.connectionError(
            requestOptions: RequestOptions(path: '/api/v1/verify'),
            reason: 'refused',
          ),
        ),
        data: Matchers.any,
      );

      await expectLater(
        api.verify(userId: 'alice', audio: audio),
        throwsA(isA<ApiException>()
            .having((e) => e.isNetworkError, 'isNetworkError', isTrue)
            .having((e) => e.message, 'message', contains('연결'))),
      );
    });

    test('본문 없는 오류에도 메시지를 만든다', () async {
      adapter.onPost('/api/v1/verify', (server) => server.reply(404, null),
          data: Matchers.any);

      await expectLater(
        api.verify(userId: 'alice', audio: audio),
        throwsA(isA<ApiException>()
            .having((e) => e.message, 'message', contains('404'))),
      );
    });
  });

  group('enroll', () {
    test('재등록 시 대체 건수를 전달한다', () async {
      adapter.onPost(
        '/api/v1/enroll',
        (server) => server.reply(200, {
          'status': 'success',
          'user_id': 'alice',
          'enrollment_id': 7,
          'replaced': 1,
          'audio': {'speech_duration_sec': 5.4},
          'elapsed_ms': 358,
        }),
        data: Matchers.any,
      );

      final result = await api.enroll(userId: 'alice', audio: audio);

      expect(result.enrollmentId, 7);
      expect(result.isReEnrollment, isTrue);
    });
  });

  group('health', () {
    test('서버 상태를 읽는다', () async {
      adapter.onGet(
        '/api/v1/health',
        (server) => server.reply(200, {
          'status': 'ok',
          'model': 'Wespeaker/wespeaker-voxceleb-resnet34-LM',
          'models_loaded': true,
          'storage': 'postgres',
          'storage_ok': true,
          'asnorm_active': true,
          'cohort_size': 310,
          'embedding_backend': 'wespeaker',
        }),
      );

      final health = await api.health();

      expect(health.isReady, isTrue);
      expect(health.asnormActive, isTrue);
      expect(health.embeddingBackend, 'wespeaker');
    });
  });
}
