/// 등록·검증 화면.
///
/// 04 §5의 데이터 플로우를 그대로 구현한다:
/// 안내 문구 → 녹음(레벨 피드백) → 전송(로딩) → 결과 표시 → 임시 파일 삭제.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../audio/recording_policy.dart';
import '../../state/voice_controller.dart';
import '../widgets/level_meter.dart';
import '../widgets/result_view.dart';

class VoiceScreen extends ConsumerStatefulWidget {
  const VoiceScreen({super.key});

  @override
  ConsumerState<VoiceScreen> createState() => _VoiceScreenState();
}

class _VoiceScreenState extends ConsumerState<VoiceScreen> {
  RecordingPurpose _purpose = RecordingPurpose.verify;
  int _promptIndex = 0;

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(voiceControllerProvider);
    final controller = ref.read(voiceControllerProvider.notifier);

    return Scaffold(
      appBar: AppBar(
        title: const Text('VoiceGuard'),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(28),
          child: _ServerBanner(),
        ),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              _PurposeSelector(
                purpose: _purpose,
                enabled: !state.isBusy,
                onChanged: (value) {
                  setState(() => _purpose = value);
                  controller.reset();
                },
              ),
              const SizedBox(height: 20),
              Expanded(
                child: Center(
                  child: SingleChildScrollView(
                    child: _Body(
                      state: state,
                      purpose: _purpose,
                      promptIndex: _promptIndex,
                    ),
                  ),
                ),
              ),
              _Controls(
                state: state,
                purpose: _purpose,
                onStart: () => controller.startRecording(_purpose),
                onStop: controller.stopAndSubmit,
                onCancel: controller.cancelRecording,
                onReset: () {
                  setState(() => _promptIndex++); // 다음 문장으로 넘긴다
                  controller.reset();
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// 서버 상태 배너.
///
/// AS-Norm이 꺼져 있거나 저장소가 인메모리면 판정 신뢰도가 낮다. 사용자가 아니라
/// 개발·운영자가 바로 알아채도록 화면에 드러낸다.
class _ServerBanner extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final health = ref.watch(serverHealthProvider);

    return health.when(
      loading: () => const SizedBox(height: 28),
      error: (_, __) => Container(
        key: const Key('server-banner-error'),
        width: double.infinity,
        color: Colors.red.shade100,
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: const Text('서버에 연결할 수 없습니다',
            textAlign: TextAlign.center, style: TextStyle(fontSize: 12)),
      ),
      data: (h) {
        final warnings = <String>[
          if (!h.asnormActive) '점수 정규화 꺼짐',
          if (h.storage == 'memory') '임시 저장소',
        ];
        if (warnings.isEmpty) return const SizedBox(height: 28);
        return Container(
          key: const Key('server-banner-warning'),
          width: double.infinity,
          color: Colors.orange.shade100,
          padding: const EdgeInsets.symmetric(vertical: 4),
          child: Text('주의: ${warnings.join(' · ')}',
              textAlign: TextAlign.center, style: const TextStyle(fontSize: 12)),
        );
      },
    );
  }
}

class _PurposeSelector extends StatelessWidget {
  const _PurposeSelector({
    required this.purpose,
    required this.enabled,
    required this.onChanged,
  });

  final RecordingPurpose purpose;
  final bool enabled;
  final ValueChanged<RecordingPurpose> onChanged;

  @override
  Widget build(BuildContext context) {
    return SegmentedButton<RecordingPurpose>(
      key: const Key('purpose-selector'),
      segments: const [
        ButtonSegment(
          value: RecordingPurpose.enroll,
          label: Text('등록'),
          icon: Icon(Icons.person_add),
        ),
        ButtonSegment(
          value: RecordingPurpose.verify,
          label: Text('인증'),
          icon: Icon(Icons.verified_user),
        ),
      ],
      selected: {purpose},
      onSelectionChanged:
          enabled ? (selection) => onChanged(selection.first) : null,
    );
  }
}

class _Body extends StatelessWidget {
  const _Body({
    required this.state,
    required this.purpose,
    required this.promptIndex,
  });

  final VoiceState state;
  final RecordingPurpose purpose;
  final int promptIndex;

  @override
  Widget build(BuildContext context) {
    switch (state.stage) {
      case VoiceStage.idle:
        return _Prompt(purpose: purpose, promptIndex: promptIndex);

      case VoiceStage.recording:
        return Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            RecordingTimer(
              elapsed: state.progress?.elapsed ?? Duration.zero,
              purpose: purpose,
            ),
            const SizedBox(height: 24),
            LevelMeter(progress: state.progress),
            const SizedBox(height: 24),
            _Prompt(purpose: purpose, promptIndex: promptIndex),
          ],
        );

      case VoiceStage.uploading:
        return const Column(
          key: Key('uploading'),
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('서버에서 분석 중입니다...'),
          ],
        );

      case VoiceStage.success:
        if (state.verifyResult != null) {
          return VerifyResultView(result: state.verifyResult!);
        }
        if (state.enrollResult != null) {
          return EnrollResultView(result: state.enrollResult!);
        }
        return const SizedBox.shrink();

      case VoiceStage.failure:
        return FailureView(
          error: state.error,
          localWarning: state.localWarning,
        );
    }
  }
}

class _Prompt extends StatelessWidget {
  const _Prompt({required this.purpose, required this.promptIndex});

  final RecordingPurpose purpose;
  final int promptIndex;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          purpose.guidance,
          key: const Key('guidance'),
          textAlign: TextAlign.center,
          style: TextStyle(color: Colors.grey.shade700),
        ),
        const SizedBox(height: 16),
        Card(
          color: Colors.blue.shade50,
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Text(
              PromptScripts.pick(purpose, promptIndex),
              key: const Key('prompt-script'),
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.titleMedium,
            ),
          ),
        ),
      ],
    );
  }
}

class _Controls extends StatelessWidget {
  const _Controls({
    required this.state,
    required this.purpose,
    required this.onStart,
    required this.onStop,
    required this.onCancel,
    required this.onReset,
  });

  final VoiceState state;
  final RecordingPurpose purpose;
  final VoidCallback onStart;
  final VoidCallback onStop;
  final VoidCallback onCancel;
  final VoidCallback onReset;

  @override
  Widget build(BuildContext context) {
    switch (state.stage) {
      case VoiceStage.idle:
        return FilledButton.icon(
          key: const Key('start-button'),
          onPressed: onStart,
          icon: const Icon(Icons.mic),
          label: Text(purpose == RecordingPurpose.enroll ? '녹음하여 등록' : '녹음하여 인증'),
          style: FilledButton.styleFrom(
            minimumSize: const Size.fromHeight(56),
          ),
        );

      case VoiceStage.recording:
        final elapsed = state.progress?.elapsed ?? Duration.zero;
        // 최소 길이를 못 채웠으면 전송 버튼을 막는다. 서버가 반려할 것을
        // 미리 알면서 보낼 이유가 없다.
        final canSubmit = elapsed >= purpose.minDuration;
        return Row(
          children: [
            Expanded(
              child: OutlinedButton(
                key: const Key('cancel-button'),
                onPressed: onCancel,
                style: OutlinedButton.styleFrom(
                    minimumSize: const Size.fromHeight(56)),
                child: const Text('취소'),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              flex: 2,
              child: FilledButton.icon(
                key: const Key('stop-button'),
                onPressed: canSubmit ? onStop : null,
                icon: const Icon(Icons.stop),
                label: const Text('완료'),
                style: FilledButton.styleFrom(
                    minimumSize: const Size.fromHeight(56)),
              ),
            ),
          ],
        );

      case VoiceStage.uploading:
        return const SizedBox(height: 56);

      case VoiceStage.success:
      case VoiceStage.failure:
        return FilledButton(
          key: const Key('retry-button'),
          onPressed: onReset,
          style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(56)),
          child: const Text('다시 하기'),
        );
    }
  }
}
