/// 녹음 정책과 반려 코드 매핑 테스트.
library;

import 'package:flutter_test/flutter_test.dart';
import 'package:voiceguard/api/errors.dart';
import 'package:voiceguard/audio/recorder.dart';
import 'package:voiceguard/audio/recording_policy.dart';

void main() {
  group('RecordingSpec', () {
    test('서버 처리 규격과 일치한다', () {
      // 16kHz/Mono로 보내야 서버 리샘플링 왜곡이 없다 (02 §1)
      expect(RecordingSpec.sampleRate, 16000);
      expect(RecordingSpec.channels, 1);
    });

    test('등록 최소 길이가 검증보다 길다', () {
      // 등록 성문은 이후 모든 검증의 기준이 되므로 더 길게 받는다
      expect(RecordingSpec.minEnrollDuration,
          greaterThan(RecordingSpec.minVerifyDuration));
    });

    test('클라이언트 최소 길이가 서버 유효 발화 하한보다 넉넉하다', () {
      // 서버 VAD 하한은 1.5초. 무음이 걷히면 녹음 길이보다 짧아지므로 여유가 필요하다
      expect(RecordingSpec.minVerifyDuration.inSeconds, greaterThan(1.5));
    });
  });

  group('DurationCheck', () {
    test('너무 짧으면 막고 필요한 길이를 알려준다', () {
      final check = DurationCheck.evaluate(
          const Duration(seconds: 1), RecordingPurpose.verify);

      expect(check.isAcceptable, isFalse);
      expect(check.message, contains('3초'));
      expect(check.message, contains('1초'));
    });

    test('등록은 더 긴 길이를 요구한다', () {
      const recorded = Duration(seconds: 4);

      expect(DurationCheck.evaluate(recorded, RecordingPurpose.verify).isAcceptable,
          isTrue);
      expect(DurationCheck.evaluate(recorded, RecordingPurpose.enroll).isAcceptable,
          isFalse,
          reason: '4초는 검증에 충분하지만 등록에는 부족하다');
    });

    test('최소 길이를 정확히 채우면 통과한다', () {
      final check = DurationCheck.evaluate(
          RecordingSpec.minVerifyDuration, RecordingPurpose.verify);

      expect(check.isAcceptable, isTrue);
    });

    test('너무 길면 막는다', () {
      final check = DurationCheck.evaluate(
          const Duration(seconds: 60), RecordingPurpose.verify);

      expect(check.isAcceptable, isFalse);
      expect(check.message, contains('이내'));
    });
  });

  group('InputLevelStatus', () {
    test('레벨을 3단계로 나눈다', () {
      expect(InputLevelStatus.fromDbfs(-60), InputLevelStatus.tooQuiet);
      expect(InputLevelStatus.fromDbfs(-20), InputLevelStatus.good);
      expect(InputLevelStatus.fromDbfs(-0.5), InputLevelStatus.clipping);
    });

    test('정상 범위에서는 경고하지 않는다', () {
      expect(InputLevelStatus.good.warning, isNull);
      expect(InputLevelStatus.tooQuiet.warning, isNotNull);
      expect(InputLevelStatus.clipping.warning, isNotNull);
    });
  });

  group('RecordingProgress', () {
    test('dBFS를 0~1 레벨로 편다', () {
      RecordingProgress at(double dbfs) => RecordingProgress(
            elapsed: Duration.zero,
            dbfs: dbfs,
            levelStatus: InputLevelStatus.fromDbfs(dbfs),
          );

      expect(at(0).normalizedLevel, 1.0);
      expect(at(-60).normalizedLevel, 0.0);
      expect(at(-30).normalizedLevel, closeTo(0.5, 1e-6));
      // 범위를 벗어난 값도 잘라낸다 — UI가 깨지면 안 된다
      expect(at(-100).normalizedLevel, 0.0);
      expect(at(10).normalizedLevel, 1.0);
    });
  });

  group('PromptScripts', () {
    test('목적별로 다른 문장을 준다', () {
      expect(PromptScripts.pick(RecordingPurpose.enroll, 0),
          isNot(PromptScripts.pick(RecordingPurpose.verify, 0)));
    });

    test('인덱스가 커져도 순환한다', () {
      // 다시 하기를 여러 번 눌러도 범위를 벗어나면 안 된다
      for (var i = 0; i < 20; i++) {
        expect(PromptScripts.pick(RecordingPurpose.enroll, i), isNotEmpty);
        expect(PromptScripts.pick(RecordingPurpose.verify, -i), isNotEmpty);
      }
    });
  });

  group('RejectionCode', () {
    test('서버 코드값과 정확히 대응한다', () {
      // 이 값들은 서버가 변경 금지 계약으로 관리한다 (server/app/core/errors.py)
      expect(RejectionCode.fromWire('speech_too_short'),
          RejectionCode.speechTooShort);
      expect(RejectionCode.fromWire('no_speech_detected'),
          RejectionCode.noSpeechDetected);
      expect(RejectionCode.fromWire('not_enrolled'), RejectionCode.notEnrolled);
      expect(RejectionCode.fromWire('model_mismatch'),
          RejectionCode.modelMismatch);
      expect(RejectionCode.fromWire('spoof_detected'),
          RejectionCode.spoofDetected);
    });

    test('모르는 코드는 unknown으로 떨어진다', () {
      // 서버가 새 사유를 추가해도 앱이 죽지 않아야 한다
      expect(RejectionCode.fromWire('brand_new_reason'), RejectionCode.unknown);
      expect(RejectionCode.fromWire(null), RejectionCode.unknown);
    });
  });

  group('ApiException', () {
    test('재녹음으로 해결되는 사유를 구분한다', () {
      const short = ApiException(
          message: '짧습니다', code: RejectionCode.speechTooShort);
      const notEnrolled = ApiException(
          message: '미등록', code: RejectionCode.notEnrolled);

      expect(short.isRetryableByReRecording, isTrue);
      expect(notEnrolled.isRetryableByReRecording, isFalse,
          reason: '다시 녹음해도 등록이 없으면 해결되지 않는다');
    });

    test('사유별 행동 안내를 제공한다', () {
      for (final code in RejectionCode.values) {
        final e = ApiException(message: 'x', code: code);
        if (code == RejectionCode.unknown) {
          // 모르는 코드는 서버의 detail을 그대로 보여준다
          expect(e.actionHint, isEmpty);
        } else {
          expect(e.actionHint, isNotEmpty,
              reason: '$code 에 대한 행동 안내가 있어야 한다');
        }
      }
    });

    test('합성 음성 차단은 직접 말하도록 안내한다', () {
      const e = ApiException(
          message: '실제 음성으로 확인되지 않았습니다.',
          code: RejectionCode.spoofDetected);

      // 탐지 점수나 모델 이름 같은 상세는 노출하지 않는다 (FR-18)
      expect(e.actionHint, contains('직접'));
      expect(e.actionHint, isNot(contains('점수')));
      // 오탐일 수 있으므로 재시도 경로는 열어둔다
      expect(e.isRetryableByReRecording, isTrue);
    });

    test('모델 불일치는 재등록을 안내한다', () {
      const e = ApiException(
          message: '호환되지 않음', code: RejectionCode.modelMismatch);

      expect(e.actionHint, contains('다시 등록'));
    });
  });
}
