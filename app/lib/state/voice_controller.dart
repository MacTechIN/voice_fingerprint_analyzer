/// 녹음 → 전송 → 결과 흐름의 상태 기계.
///
/// UI는 이 상태만 보고 그린다. 녹음·네트워크·오류 처리가 화면 코드에 흩어지면
/// 흐름을 따라가기 어렵고, 임시 파일 삭제 같은 뒷정리를 빠뜨리기 쉽다.
library;

import 'dart:async';
import 'dart:io';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/client.dart';
import '../api/errors.dart';
import '../api/models.dart';
import '../audio/recorder.dart';
import '../audio/recording_policy.dart';

/// 흐름 단계.
enum VoiceStage {
  idle,
  recording,
  uploading,
  success,
  failure,
}

/// 화면이 그리는 데 필요한 전부.
class VoiceState {
  const VoiceState({
    this.stage = VoiceStage.idle,
    this.purpose = RecordingPurpose.verify,
    this.progress,
    this.enrollResult,
    this.verifyResult,
    this.error,
    this.localWarning,
  });

  final VoiceStage stage;
  final RecordingPurpose purpose;

  /// 녹음 중일 때만 채워진다.
  final RecordingProgress? progress;

  final EnrollResult? enrollResult;
  final VerifyResult? verifyResult;

  /// 서버가 반려했거나 네트워크가 실패한 경우.
  final ApiException? error;

  /// 서버에 보내기 전 앱이 막은 경우 (녹음이 너무 짧음 등).
  final String? localWarning;

  bool get isBusy =>
      stage == VoiceStage.recording || stage == VoiceStage.uploading;

  VoiceState copyWith({
    VoiceStage? stage,
    RecordingPurpose? purpose,
    RecordingProgress? progress,
    EnrollResult? enrollResult,
    VerifyResult? verifyResult,
    ApiException? error,
    String? localWarning,
    bool clearProgress = false,
    bool clearResults = false,
    bool clearError = false,
  }) {
    return VoiceState(
      stage: stage ?? this.stage,
      purpose: purpose ?? this.purpose,
      progress: clearProgress ? null : (progress ?? this.progress),
      enrollResult: clearResults ? null : (enrollResult ?? this.enrollResult),
      verifyResult: clearResults ? null : (verifyResult ?? this.verifyResult),
      error: clearError ? null : (error ?? this.error),
      localWarning: clearError ? null : (localWarning ?? this.localWarning),
    );
  }
}

class VoiceController extends StateNotifier<VoiceState> {
  VoiceController({
    required AudioRecorderService recorder,
    required VoiceGuardApi api,
    required this.userId,
  })  : _recorder = recorder,
        _api = api,
        super(const VoiceState());

  final AudioRecorderService _recorder;
  final VoiceGuardApi _api;
  final String userId;

  StreamSubscription<RecordingProgress>? _progressSub;

  /// 녹음 시작.
  Future<void> startRecording(RecordingPurpose purpose) async {
    if (state.isBusy) return;

    if (!await _recorder.hasPermission()) {
      state = state.copyWith(
        stage: VoiceStage.failure,
        error: const ApiException(
          message: '마이크 권한이 필요합니다. 설정에서 권한을 허용해주세요.',
        ),
        clearResults: true,
      );
      return;
    }

    state = VoiceState(stage: VoiceStage.recording, purpose: purpose);

    await _progressSub?.cancel();
    _progressSub = _recorder.start().listen(
      (progress) {
        // 최대 길이를 넘기면 자동으로 마감한다. 사용자가 버튼에서 손을 떼지
        // 않아도 무한정 녹음되지 않는다.
        if (progress.elapsed >= RecordingSpec.maxDuration) {
          unawaited(stopAndSubmit());
          return;
        }
        state = state.copyWith(progress: progress);
      },
      onError: (Object e) {
        state = state.copyWith(
          stage: VoiceStage.failure,
          error: ApiException(message: '녹음을 시작할 수 없습니다: $e'),
          clearProgress: true,
        );
      },
    );
  }

  /// 녹음을 마치고 서버로 보낸다.
  Future<void> stopAndSubmit() async {
    if (state.stage != VoiceStage.recording) return;

    await _progressSub?.cancel();
    _progressSub = null;

    final recording = await _recorder.stop();
    if (recording == null) {
      state = state.copyWith(
        stage: VoiceStage.failure,
        error: const ApiException(message: '녹음 파일을 만들지 못했습니다.'),
        clearProgress: true,
      );
      return;
    }

    // 서버로 보내기 전에 길이를 검사한다. 짧은 녹음을 그대로 보내면 왕복 후에야
    // "더 길게 말하세요"를 듣게 된다.
    final check = DurationCheck.evaluate(recording.duration, state.purpose);
    if (!check.isAcceptable) {
      await deleteTempAudio(recording.file);
      state = state.copyWith(
        stage: VoiceStage.failure,
        localWarning: check.message,
        clearProgress: true,
      );
      return;
    }

    state = state.copyWith(stage: VoiceStage.uploading, clearProgress: true);
    await _upload(recording.file);
  }

  Future<void> _upload(File file) async {
    try {
      if (state.purpose == RecordingPurpose.enroll) {
        final result = await _api.enroll(userId: userId, audio: file);
        state = state.copyWith(stage: VoiceStage.success, enrollResult: result);
      } else {
        final result = await _api.verify(userId: userId, audio: file);
        state = state.copyWith(stage: VoiceStage.success, verifyResult: result);
      }
    } on ApiException catch (e) {
      state = state.copyWith(stage: VoiceStage.failure, error: e);
    } catch (e) {
      state = state.copyWith(
        stage: VoiceStage.failure,
        error: ApiException(message: '알 수 없는 오류가 발생했습니다: $e'),
      );
    } finally {
      // 성공이든 실패든 임시 파일은 남기지 않는다 (FR-02, 04 §6).
      // 생체정보를 기기에 남겨두지 않는 것이 원칙이다.
      await deleteTempAudio(file);
    }
  }

  /// 녹음 취소 — 서버로 보내지 않는다.
  Future<void> cancelRecording() async {
    await _progressSub?.cancel();
    _progressSub = null;
    await _recorder.cancel();
    state = const VoiceState();
  }

  /// 결과 화면에서 처음으로 돌아간다.
  void reset() {
    state = VoiceState(purpose: state.purpose);
  }

  @override
  void dispose() {
    unawaited(_progressSub?.cancel());
    unawaited(_recorder.dispose());
    super.dispose();
  }
}

// --- Providers ---

/// 사용자 ID. 실제 서비스에서는 로그인 세션에서 온다.
final userIdProvider = StateProvider<String>((ref) => 'demo-user');

final apiProvider = Provider<VoiceGuardApi>((ref) => VoiceGuardApi());

final recorderProvider = Provider<AudioRecorderService>((ref) {
  final recorder = DeviceAudioRecorder();
  ref.onDispose(recorder.dispose);
  return recorder;
});

final voiceControllerProvider =
    StateNotifierProvider<VoiceController, VoiceState>((ref) {
  return VoiceController(
    recorder: ref.watch(recorderProvider),
    api: ref.watch(apiProvider),
    userId: ref.watch(userIdProvider),
  );
});

/// 서버 상태 — 화면 상단 배너에 쓴다.
final serverHealthProvider = FutureProvider<ServerHealth>((ref) async {
  return ref.watch(apiProvider).health();
});
