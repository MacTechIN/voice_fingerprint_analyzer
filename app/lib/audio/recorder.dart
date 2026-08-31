/// 마이크 녹음 서비스.
///
/// 앱의 책임은 **분석 품질이 보장되는 오디오를 수집해 전달하는 것**까지다
/// (04 §1). 무거운 머신러닝 라이브러리는 일절 포함하지 않는다.
library;

import 'dart:async';
import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import 'recording_policy.dart';

/// 녹음 진행 상태 — UI가 파형·타이머·경고를 그리는 데 쓴다.
class RecordingProgress {
  const RecordingProgress({
    required this.elapsed,
    required this.dbfs,
    required this.levelStatus,
  });

  final Duration elapsed;
  final double dbfs;
  final InputLevelStatus levelStatus;

  /// 0~1로 정규화한 레벨. 파형·미터 그리기에 쓴다.
  ///
  /// dBFS는 로그 척도라 -60~0을 선형으로 펴서 시각화한다.
  double get normalizedLevel {
    const floor = -60.0;
    if (dbfs <= floor) return 0;
    if (dbfs >= 0) return 1;
    return (dbfs - floor) / -floor;
  }
}

/// 녹음 결과.
class RecordingResult {
  const RecordingResult({required this.file, required this.duration});

  final File file;
  final Duration duration;
}

/// 녹음기 인터페이스.
///
/// 실제 마이크 접근은 테스트에서 쓸 수 없으므로 인터페이스로 분리한다.
/// 위젯 테스트는 [FakeRecorder]를 주입해 UI 흐름만 검증한다.
abstract class AudioRecorderService {
  /// 마이크 권한 확인·요청.
  Future<bool> hasPermission();

  /// 녹음 시작. 진행 상태 스트림을 반환한다.
  Stream<RecordingProgress> start();

  /// 녹음 중지. 파일과 실제 길이를 반환한다.
  Future<RecordingResult?> stop();

  /// 녹음 취소 — 파일을 남기지 않는다.
  Future<void> cancel();

  bool get isRecording;

  Future<void> dispose();
}

/// `record` 패키지 기반 구현.
class DeviceAudioRecorder implements AudioRecorderService {
  DeviceAudioRecorder({AudioRecorder? recorder})
      : _recorder = recorder ?? AudioRecorder();

  final AudioRecorder _recorder;

  Timer? _ticker;
  StreamController<RecordingProgress>? _progress;
  DateTime? _startedAt;
  String? _path;

  @override
  bool get isRecording => _startedAt != null;

  @override
  Future<bool> hasPermission() => _recorder.hasPermission();

  @override
  Stream<RecordingProgress> start() {
    final controller = StreamController<RecordingProgress>.broadcast(
      onCancel: () {}, // 스트림 구독이 끊겨도 녹음은 stop()까지 계속된다
    );
    _progress = controller;

    unawaited(_beginRecording(controller));
    return controller.stream;
  }

  Future<void> _beginRecording(
      StreamController<RecordingProgress> controller) async {
    try {
      final dir = await getTemporaryDirectory();
      final path =
          '${dir.path}/vg_${DateTime.now().millisecondsSinceEpoch}.wav';
      _path = path;

      // WAV/PCM 16bit로 직접 받는다. 서버가 그대로 디코딩할 수 있어 변환이 없다.
      await _recorder.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: RecordingSpec.sampleRate,
          numChannels: RecordingSpec.channels,
        ),
        path: path,
      );
      _startedAt = DateTime.now();

      // 진폭 폴링. 100ms면 사용자가 레벨 변화를 즉시 느끼면서도 부담이 없다.
      _ticker = Timer.periodic(const Duration(milliseconds: 100), (_) async {
        if (_startedAt == null || controller.isClosed) return;
        try {
          final amplitude = await _recorder.getAmplitude();
          final dbfs = amplitude.current;
          controller.add(RecordingProgress(
            elapsed: DateTime.now().difference(_startedAt!),
            dbfs: dbfs,
            levelStatus: InputLevelStatus.fromDbfs(dbfs),
          ));
        } catch (_) {
          // 진폭 조회 실패가 녹음을 멈추게 해서는 안 된다
        }
      });
    } catch (e) {
      if (!controller.isClosed) controller.addError(e);
      await _cleanup();
    }
  }

  @override
  Future<RecordingResult?> stop() async {
    final startedAt = _startedAt;
    final path = _path;
    final duration =
        startedAt == null ? Duration.zero : DateTime.now().difference(startedAt);

    await _cleanup();
    final recordedPath = await _recorder.stop();

    final resolved = recordedPath ?? path;
    if (resolved == null) return null;

    final file = File(resolved);
    if (!file.existsSync()) return null;

    return RecordingResult(file: file, duration: duration);
  }

  @override
  Future<void> cancel() async {
    await _cleanup();
    await _recorder.cancel();
    final path = _path;
    if (path != null) {
      final file = File(path);
      if (file.existsSync()) {
        try {
          await file.delete();
        } catch (_) {
          // 임시 파일 삭제 실패는 치명적이지 않다
        }
      }
    }
    _path = null;
  }

  Future<void> _cleanup() async {
    _ticker?.cancel();
    _ticker = null;
    _startedAt = null;
    await _progress?.close();
    _progress = null;
  }

  @override
  Future<void> dispose() async {
    await _cleanup();
    await _recorder.dispose();
  }
}

/// 임시 오디오 파일 정리.
///
/// 전송 성공 즉시 삭제한다 (FR-02, 04 §6). 생체정보를 기기에 남기지 않는다.
Future<void> deleteTempAudio(File file) async {
  try {
    if (file.existsSync()) await file.delete();
  } catch (_) {
    // 삭제 실패가 사용자 흐름을 막아서는 안 된다. 다음 정리 때 지워진다.
  }
}
