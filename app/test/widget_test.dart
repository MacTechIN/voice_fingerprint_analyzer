/// 화면 렌더링 테스트.
///
/// 04 §5의 데이터 플로우가 화면에 그대로 드러나는지 확인한다.
library;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:voiceguard/api/errors.dart';
import 'package:voiceguard/api/models.dart';
import 'package:voiceguard/audio/recorder.dart';
import 'package:voiceguard/audio/recording_policy.dart';
import 'package:voiceguard/ui/widgets/level_meter.dart';
import 'package:voiceguard/ui/widgets/result_view.dart';

Widget _wrap(Widget child) =>
    MaterialApp(home: Scaffold(body: Center(child: child)));

void main() {
  group('LevelMeter', () {
    testWidgets('소리가 작으면 경고를 띄운다', (tester) async {
      await tester.pumpWidget(_wrap(const LevelMeter(
        progress: RecordingProgress(
          elapsed: Duration(seconds: 1),
          dbfs: -55,
          levelStatus: InputLevelStatus.tooQuiet,
        ),
      )));

      expect(find.byKey(const Key('level-warning')), findsOneWidget);
      expect(find.textContaining('가까이'), findsOneWidget);
    });

    testWidgets('클리핑이면 다른 경고를 띄운다', (tester) async {
      await tester.pumpWidget(_wrap(const LevelMeter(
        progress: RecordingProgress(
          elapsed: Duration(seconds: 1),
          dbfs: -0.5,
          levelStatus: InputLevelStatus.clipping,
        ),
      )));

      expect(find.textContaining('떨어져'), findsOneWidget);
    });

    testWidgets('정상 레벨에서는 경고가 없다', (tester) async {
      await tester.pumpWidget(_wrap(const LevelMeter(
        progress: RecordingProgress(
          elapsed: Duration(seconds: 1),
          dbfs: -20,
          levelStatus: InputLevelStatus.good,
        ),
      )));

      expect(find.byKey(const Key('level-warning')), findsNothing);
    });
  });

  group('RecordingTimer', () {
    testWidgets('최소 길이 미달을 알린다', (tester) async {
      await tester.pumpWidget(_wrap(const RecordingTimer(
        elapsed: Duration(milliseconds: 1500),
        purpose: RecordingPurpose.verify,
      )));

      expect(find.text('1.5초'), findsOneWidget);
      expect(find.textContaining('3초 이상 필요'), findsOneWidget);
    });

    testWidgets('충분히 녹음되면 안내가 바뀐다', (tester) async {
      await tester.pumpWidget(_wrap(const RecordingTimer(
        elapsed: Duration(seconds: 4),
        purpose: RecordingPurpose.verify,
      )));

      expect(find.textContaining('충분합니다'), findsOneWidget);
    });
  });

  group('VerifyResultView', () {
    VerifyResult result({
      bool verified = true,
      double probability = 95.4,
      String method = 'as_norm',
    }) =>
        VerifyResult(
          userId: 'alice',
          isVerified: verified,
          matchProbability: probability,
          rawCosine: 0.8,
          normalizedScore: method == 'as_norm' ? 9.1 : null,
          threshold: 2.9673,
          scoringMethod: method,
          comparedEnrollments: 1,
          audio: const AudioInfo(
            durationSec: 5,
            speechDurationSec: 3.5,
            speechRatio: 0.7,
            sampleRate: 16000,
            segments: [],
          ),
          elapsedMs: 116,
        );

    testWidgets('일치도와 판정을 보여준다', (tester) async {
      await tester.pumpWidget(_wrap(VerifyResultView(result: result())));
      await tester.pumpAndSettle();

      expect(find.byKey(const Key('match-probability')), findsOneWidget);
      expect(find.text('95.4%'), findsOneWidget);
      expect(find.text('본인 확인됨'), findsOneWidget);
      // "확률"이 아니라 "일치도"로 표기한다 — 실제로 확률이 아니기 때문이다
      expect(find.text('일치도'), findsOneWidget);
      expect(find.textContaining('확률'), findsNothing);
    });

    testWidgets('실패 판정을 구분해 보여준다', (tester) async {
      await tester.pumpWidget(
          _wrap(VerifyResultView(result: result(verified: false, probability: 12))));
      await tester.pumpAndSettle();

      expect(find.text('본인 확인 실패'), findsOneWidget);
    });

    testWidgets('정규화 없이 판정되면 경고를 드러낸다', (tester) async {
      await tester.pumpWidget(
          _wrap(VerifyResultView(result: result(method: 'raw_cosine'))));
      await tester.pumpAndSettle();

      // 서버 코호트가 없는 상태를 숨기지 않는다 — 판정 신뢰도가 낮다
      expect(find.byKey(const Key('fallback-warning')), findsOneWidget);
    });

    testWidgets('정규화가 적용되면 경고가 없다', (tester) async {
      await tester.pumpWidget(_wrap(VerifyResultView(result: result())));
      await tester.pumpAndSettle();

      expect(find.byKey(const Key('fallback-warning')), findsNothing);
    });
  });

  group('EnrollResultView', () {
    testWidgets('첫 등록과 재등록 문구가 다르다', (tester) async {
      const audio = AudioInfo(
        durationSec: 6,
        speechDurationSec: 5.4,
        speechRatio: 0.9,
        sampleRate: 16000,
        segments: [],
      );

      await tester.pumpWidget(_wrap(const EnrollResultView(
        result: EnrollResult(
          userId: 'alice',
          enrollmentId: 1,
          replaced: 0,
          audio: audio,
          elapsedMs: 358,
        ),
      )));
      expect(find.text('음성이 등록되었습니다'), findsOneWidget);

      await tester.pumpWidget(_wrap(const EnrollResultView(
        result: EnrollResult(
          userId: 'alice',
          enrollmentId: 2,
          replaced: 1,
          audio: audio,
          elapsedMs: 358,
        ),
      )));
      expect(find.text('음성이 다시 등록되었습니다'), findsOneWidget);
      expect(find.textContaining('기존 등록 1건'), findsOneWidget);
    });
  });

  group('FailureView', () {
    testWidgets('서버 문구와 행동 안내를 함께 보여준다', (tester) async {
      await tester.pumpWidget(_wrap(const FailureView(
        error: ApiException(
          message: '유효 발화가 너무 짧습니다 (0.4초).',
          code: RejectionCode.speechTooShort,
        ),
      )));

      // 서버는 "무엇이 잘못됐는지", 앱은 "무엇을 하면 되는지"를 알려준다
      expect(find.textContaining('0.4초'), findsOneWidget);
      expect(find.byKey(const Key('failure-hint')), findsOneWidget);
      expect(find.textContaining('길게 말해주세요'), findsOneWidget);
    });

    testWidgets('앱이 막은 경우 로컬 경고만 보여준다', (tester) async {
      await tester.pumpWidget(
          _wrap(const FailureView(localWarning: '3초 이상 녹음해주세요. (현재 1초)')));

      expect(find.textContaining('3초 이상'), findsOneWidget);
      expect(find.byKey(const Key('failure-hint')), findsNothing);
    });

    testWidgets('모르는 사유는 서버 문구만 보여준다', (tester) async {
      await tester.pumpWidget(_wrap(const FailureView(
        error: ApiException(
          message: '실제 음성 확인에 실패했습니다.',
          code: RejectionCode.unknown,
        ),
      )));

      expect(find.textContaining('실제 음성'), findsOneWidget);
      expect(find.byKey(const Key('failure-hint')), findsNothing);
    });
  });
}
