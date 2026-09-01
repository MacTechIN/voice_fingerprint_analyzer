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
  const RecordingResult({
    required this.file,
    required this.duration,
    this.format = UploadFormat.wav,
  });

  final File file;
  final Duration duration;

  /// 실제로 녹음된 포맷. 업로드 시 파일명·MIME을 맞추는 데 쓴다.
  final UploadFormat format;
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

  /// 이번 녹음에 쓸 포맷. 플랫폼이 FLAC을 지원하면 FLAC, 아니면 WAV.
  Future<UploadFormat> resolveFormat();

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

  /// 협상된 업로드 포맷. 한 번 정해지면 프로세스 수명 동안 유지한다 —
  /// 플랫폼의 인코더 지원 여부는 실행 중에 바뀌지 않는다.
  UploadFormat? _format;

  @override
  bool get isRecording => _startedAt != null;

  @override
  Future<bool> hasPermission() => _recorder.hasPermission();

  @override
  Future<UploadFormat> resolveFormat() async {
    final cached = _format;
    if (cached != null) return cached;

    // FLAC은 무손실이라 임베딩에 영향이 없으면서 업로드가 WAV의 62%다.
    // 다만 플랫폼·OS 버전에 따라 인코더가 없을 수 있으므로 **런타임에 물어보고**
    // 없으면 WAV로 내려간다. 지원하지 않는 포맷으로 녹음을 시도하면 녹음 자체가
    // 실패하는데, 그건 업로드가 좀 큰 것보다 훨씬 나쁘다.
    var resolved = UploadFormat.wav;
    try {
      if (await _recorder.isEncoderSupported(AudioEncoder.flac)) {
        resolved = UploadFormat.flac;
      }
    } catch (_) {
      // 조회 자체가 실패하면 안전한 쪽(WAV)을 쓴다
    }
    _format = resolved;
    return resolved;
  }

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
      final format = await resolveFormat();
      final dir = await getTemporaryDirectory();
      final path =
          '${dir.path}/vg_${DateTime.now().millisecondsSinceEpoch}.${format.extension}';
      _path = path;

      // 16kHz/Mono로 직접 받는다. 서버 규격과 같아 리샘플링 왜곡이 없고,
      // 서버는 두 포맷 모두 그대로 디코딩한다.
      await _recorder.start(
        RecordConfig(
          encoder: format == UploadFormat.flac
              ? AudioEncoder.flac
              : AudioEncoder.wav,
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

    return RecordingResult(
      file: file,
      duration: duration,
      format: _format ?? UploadFormat.wav,
    );
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
