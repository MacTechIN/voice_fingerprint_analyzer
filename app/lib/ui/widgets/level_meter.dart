/// 녹음 중 입력 레벨 시각 피드백 (FR-01).
library;

import 'package:flutter/material.dart';

import '../../audio/recorder.dart';
import '../../audio/recording_policy.dart';

/// 실시간 입력 레벨 바.
///
/// 파형 대신 레벨 미터를 쓴 이유: 사용자가 판단해야 할 것은 "파형 모양"이 아니라
/// **소리가 충분히 크고 클리핑되지 않는가**이며, 그건 미터가 더 직접적으로 보여준다.
class LevelMeter extends StatelessWidget {
  const LevelMeter({super.key, required this.progress});

  final RecordingProgress? progress;

  @override
  Widget build(BuildContext context) {
    final level = progress?.normalizedLevel ?? 0;
    final status = progress?.levelStatus ?? InputLevelStatus.good;

    final color = switch (status) {
      InputLevelStatus.clipping => Colors.red,
      InputLevelStatus.tooQuiet => Colors.orange,
      InputLevelStatus.good => Colors.green,
    };

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: LinearProgressIndicator(
            value: level,
            minHeight: 12,
            backgroundColor: Colors.grey.shade300,
            valueColor: AlwaysStoppedAnimation<Color>(color),
          ),
        ),
        const SizedBox(height: 8),
        if (status.warning != null)
          Text(
            status.warning!,
            key: const Key('level-warning'),
            textAlign: TextAlign.center,
            style: TextStyle(color: color, fontWeight: FontWeight.w500),
          ),
      ],
    );
  }
}

/// 녹음 경과 시간과 최소 길이 충족 여부.
class RecordingTimer extends StatelessWidget {
  const RecordingTimer({
    super.key,
    required this.elapsed,
    required this.purpose,
  });

  final Duration elapsed;
  final RecordingPurpose purpose;

  @override
  Widget build(BuildContext context) {
    final minimum = purpose.minDuration;
    final isEnough = elapsed >= minimum;
    final seconds = (elapsed.inMilliseconds / 1000).toStringAsFixed(1);

    return Column(
      children: [
        Text(
          '$seconds초',
          key: const Key('recording-timer'),
          style: Theme.of(context).textTheme.headlineMedium,
        ),
        const SizedBox(height: 4),
        Text(
          isEnough ? '충분합니다 — 버튼을 놓으면 전송됩니다' : '${minimum.inSeconds}초 이상 필요합니다',
          key: const Key('duration-hint'),
          style: TextStyle(
            color: isEnough ? Colors.green.shade700 : Colors.grey.shade600,
          ),
        ),
      ],
    );
  }
}
