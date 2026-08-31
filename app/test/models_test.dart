/// 서버 응답 파싱 테스트.
///
/// 서버는 확장 가능 스키마를 쓰며 Phase 7~8에서 필드가 추가된다. 앱이 그때마다
/// 깨지면 서버 배포에 앱 배포가 묶이므로, **알 수 없는 필드를 무시하는지**가
/// 이 테스트의 핵심이다 (01 §2 Tolerant Reader).
library;

import 'package:flutter_test/flutter_test.dart';
import 'package:voiceguard/api/models.dart';

/// 서버가 실제로 보내는 검증 응답 (server/README 예시).
const Map<String, dynamic> _verifyJson = {
  'status': 'success',
  'user_id': 'alice',
  'is_verified': true,
  'match_probability': 100.0,
  'raw_cosine': 0.7722,
  'normalized_score': 9.1427,
  'scoring_method': 'as_norm',
  'threshold': 2.9673,
  'compared_enrollments': 1,
  'audio': {
    'duration_sec': 7.0,
    'speech_duration_sec': 3.068,
    'speech_ratio': 0.438,
    'sample_rate': 16000,
    'source_sample_rate': 16000,
    'source_channels': 1,
    'segments': [
      {'start': 1.986, 'end': 5.054}
    ],
  },
  'elapsed_ms': 115.7,
};

void main() {
  group('VerifyResult', () {
    test('서버 응답을 그대로 파싱한다', () {
      final result = VerifyResult.fromJson(_verifyJson);

      expect(result.userId, 'alice');
      expect(result.isVerified, isTrue);
      expect(result.matchProbability, 100.0);
      expect(result.rawCosine, closeTo(0.7722, 1e-6));
      expect(result.normalizedScore, closeTo(9.1427, 1e-6));
      expect(result.scoringMethod, 'as_norm');
      expect(result.threshold, closeTo(2.9673, 1e-6));
      expect(result.comparedEnrollments, 1);
      expect(result.audio.speechDurationSec, closeTo(3.068, 1e-6));
      expect(result.audio.segments, hasLength(1));
      expect(result.audio.segments.first.duration, closeTo(3.068, 1e-3));
    });

    test('알 수 없는 필드가 추가돼도 깨지지 않는다', () {
      // Phase 8에서 spoof_score가 최상위에 추가되는 상황
      final future = {
        ..._verifyJson,
        'spoof_score': 0.02,
        'liveness': {'passed': true, 'model': 'aasist-l'},
        'brand_new_field': ['whatever'],
      };

      final result = VerifyResult.fromJson(future);

      expect(result.isVerified, isTrue);
      expect(result.matchProbability, 100.0);
    });

    test('AS-Norm 폴백 응답을 처리한다', () {
      // 서버 코호트가 없으면 normalized_score가 null로 온다
      final fallback = {
        ..._verifyJson,
        'normalized_score': null,
        'scoring_method': 'raw_cosine',
        'threshold': 0.3767,
      };

      final result = VerifyResult.fromJson(fallback);

      expect(result.normalizedScore, isNull);
      expect(result.isFallbackScoring, isTrue,
          reason: '정규화 없이 판정된 사실을 UI가 드러낼 수 있어야 한다');
    });

    test('as_norm이면 폴백이 아니다', () {
      expect(VerifyResult.fromJson(_verifyJson).isFallbackScoring, isFalse);
    });

    test('정수로 온 숫자도 처리한다', () {
      // JSON에서 100.0이 100으로 직렬화되면 Dart는 int로 받는다.
      // `as double` 캐스팅이었다면 여기서 런타임에 터진다.
      final intish = {
        ..._verifyJson,
        'match_probability': 100,
        'elapsed_ms': 116,
        'threshold': 3,
      };

      final result = VerifyResult.fromJson(intish);

      expect(result.matchProbability, 100.0);
      expect(result.elapsedMs, 116.0);
      expect(result.threshold, 3.0);
    });

    test('필드가 빠져도 기본값으로 버틴다', () {
      final result = VerifyResult.fromJson(const {'user_id': 'bob'});

      expect(result.userId, 'bob');
      expect(result.isVerified, isFalse);
      expect(result.matchProbability, 0);
      expect(result.audio.segments, isEmpty);
    });
  });

  group('EnrollResult', () {
    test('첫 등록과 재등록을 구분한다', () {
      final first = EnrollResult.fromJson(const {
        'user_id': 'alice',
        'enrollment_id': 1,
        'replaced': 0,
        'audio': {'speech_duration_sec': 5.2},
        'elapsed_ms': 358.0,
      });
      final again = EnrollResult.fromJson(const {
        'user_id': 'alice',
        'enrollment_id': 2,
        'replaced': 1,
        'audio': {'speech_duration_sec': 5.2},
        'elapsed_ms': 358.0,
      });

      expect(first.isReEnrollment, isFalse);
      expect(again.isReEnrollment, isTrue);
      expect(again.replaced, 1);
    });
  });

  group('ServerHealth', () {
    test('준비 상태를 판정한다', () {
      final healthy = ServerHealth.fromJson(const {
        'status': 'ok',
        'model': 'Wespeaker/wespeaker-voxceleb-resnet34-LM',
        'models_loaded': true,
        'storage': 'postgres',
        'storage_ok': true,
        'asnorm_active': true,
        'cohort_size': 310,
        'embedding_backend': 'wespeaker',
      });

      expect(healthy.isReady, isTrue);
      expect(healthy.asnormActive, isTrue);
      expect(healthy.cohortSize, 310);
    });

    test('저장소가 죽으면 준비되지 않은 것으로 본다', () {
      final broken = ServerHealth.fromJson(const {
        'status': 'ok',
        'models_loaded': true,
        'storage_ok': false,
      });

      expect(broken.isReady, isFalse);
    });
  });
}
