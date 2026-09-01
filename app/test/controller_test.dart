/// 녹음 → 전송 → 결과 흐름 테스트.
///
/// 마이크는 테스트에서 쓸 수 없으므로 [FakeRecorder]를 주입한다. 검증 대상은
/// 오디오 캡처 자체가 아니라 **상태 전이와 뒷정리**다 — 특히 임시 파일이
/// 항상 삭제되는지 (FR-02).
library;

import 'dart:async';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http_mock_adapter/http_mock_adapter.dart';
import 'package:voiceguard/api/client.dart';
import 'package:voiceguard/api/errors.dart';
import 'package:voiceguard/audio/recorder.dart';
import 'package:voiceguard/audio/recording_policy.dart';
import 'package:voiceguard/state/voice_controller.dart';

class FakeRecorder implements AudioRecorderService {
  FakeRecorder({
    this.permitted = true,
    this.duration = const Duration(seconds: 4),
    this.failToProduceFile = false,
    this.format = UploadFormat.flac,
  });

  final bool permitted;
  final Duration duration;
  final bool failToProduceFile;
  final UploadFormat format;

  File? lastFile;
  bool cancelled = false;
  bool disposed = false;

  final _controller = StreamController<RecordingProgress>.broadcast();
  bool _recording = false;

  @override
  bool get isRecording => _recording;

  @override
  Future<bool> hasPermission() async => permitted;

  @override
  Future<UploadFormat> resolveFormat() async => format;

  @override
  Stream<RecordingProgress> start() {
    _recording = true;
    return _controller.stream;
  }

  /// 테스트에서 진행 상태를 주입한다.
  void emit(Duration elapsed, {double dbfs = -20}) {
    _controller.add(RecordingProgress(
      elapsed: elapsed,
      dbfs: dbfs,
      levelStatus: InputLevelStatus.fromDbfs(dbfs),
    ));
  }

  @override
  Future<RecordingResult?> stop() async {
    _recording = false;
    if (failToProduceFile) return null;

    final file = File(
        '${Directory.systemTemp.path}/vg_fake_${DateTime.now().microsecondsSinceEpoch}.wav')
      ..writeAsBytesSync(List<int>.filled(64, 0));
    lastFile = file;
    return RecordingResult(file: file, duration: duration, format: format);
  }

  @override
  Future<void> cancel() async {
    _recording = false;
    cancelled = true;
  }

  @override
  Future<void> dispose() async {
    disposed = true;
    await _controller.close();
  }
}

void main() {
  late Dio dio;
  late DioAdapter adapter;
  late VoiceGuardApi api;

  setUp(() {
    dio = Dio(BaseOptions(
      baseUrl: 'http://test.local',
      validateStatus: (status) => status != null && status < 500,
    ));
    adapter = DioAdapter(dio: dio);
    api = VoiceGuardApi(dio: dio);
  });

  VoiceController makeController(FakeRecorder recorder) =>
      VoiceController(recorder: recorder, api: api, userId: 'alice');

  void mockVerifySuccess() {
    adapter.onPost(
      '/api/v1/verify',
      (server) => server.reply(200, {
        'status': 'success',
        'user_id': 'alice',
        'is_verified': true,
        'match_probability': 95.4,
        'raw_cosine': 0.8,
        'normalized_score': 9.1,
        'scoring_method': 'as_norm',
        'threshold': 2.9673,
        'compared_enrollments': 1,
        'audio': {'speech_duration_sec': 3.5},
        'elapsed_ms': 116,
      }),
      data: Matchers.any,
    );
  }

  group('녹음 시작', () {
    test('권한이 없으면 실패로 끝난다', () async {
      final recorder = FakeRecorder(permitted: false);
      final controller = makeController(recorder);

      await controller.startRecording(RecordingPurpose.verify);

      expect(controller.state.stage, VoiceStage.failure);
      expect(controller.state.error?.message, contains('권한'));
    });

    test('권한이 있으면 녹음 상태로 간다', () async {
      final controller = makeController(FakeRecorder());

      await controller.startRecording(RecordingPurpose.verify);

      expect(controller.state.stage, VoiceStage.recording);
      expect(controller.state.purpose, RecordingPurpose.verify);
    });

    test('진행 상태가 상태에 반영된다', () async {
      final recorder = FakeRecorder();
      final controller = makeController(recorder);
      await controller.startRecording(RecordingPurpose.verify);

      recorder.emit(const Duration(seconds: 2), dbfs: -50);
      await Future<void>.delayed(Duration.zero);

      expect(controller.state.progress?.elapsed, const Duration(seconds: 2));
      expect(controller.state.progress?.levelStatus, InputLevelStatus.tooQuiet);
    });
  });

  group('전송', () {
    test('정상 흐름은 성공 결과를 남긴다', () async {
      mockVerifySuccess();
      final recorder = FakeRecorder(duration: const Duration(seconds: 4));
      final controller = makeController(recorder);

      await controller.startRecording(RecordingPurpose.verify);
      await controller.stopAndSubmit();

      expect(controller.state.stage, VoiceStage.success);
      expect(controller.state.verifyResult?.isVerified, isTrue);
      expect(controller.state.verifyResult?.matchProbability, 95.4);
    });

    test('전송 성공 후 임시 파일을 지운다', () async {
      mockVerifySuccess();
      final recorder = FakeRecorder(duration: const Duration(seconds: 4));
      final controller = makeController(recorder);

      await controller.startRecording(RecordingPurpose.verify);
      await controller.stopAndSubmit();

      // 생체정보를 기기에 남기지 않는다 (FR-02, 04 §6)
      expect(recorder.lastFile!.existsSync(), isFalse);
    });

    test('전송 실패해도 임시 파일을 지운다', () async {
      adapter.onPost(
        '/api/v1/verify',
        (server) => server.reply(422, {
          'status': 'error',
          'code': 'no_speech_detected',
          'detail': '음성이 감지되지 않았습니다.',
        }),
        data: Matchers.any,
      );
      final recorder = FakeRecorder(duration: const Duration(seconds: 4));
      final controller = makeController(recorder);

      await controller.startRecording(RecordingPurpose.verify);
      await controller.stopAndSubmit();

      expect(controller.state.stage, VoiceStage.failure);
      expect(controller.state.error?.code, RejectionCode.noSpeechDetected);
      expect(recorder.lastFile!.existsSync(), isFalse,
          reason: '실패 경로에서도 파일이 남으면 안 된다');
    });

    test('너무 짧으면 서버로 보내지 않는다', () async {
      // 서버 호출을 아예 등록하지 않는다 — 호출되면 DioAdapter가 터진다
      final recorder = FakeRecorder(duration: const Duration(seconds: 1));
      final controller = makeController(recorder);

      await controller.startRecording(RecordingPurpose.verify);
      await controller.stopAndSubmit();

      expect(controller.state.stage, VoiceStage.failure);
      expect(controller.state.localWarning, contains('3초'));
      expect(controller.state.error, isNull,
          reason: '서버 반려가 아니라 앱이 막은 것이다');
      expect(recorder.lastFile!.existsSync(), isFalse);
    });

    test('등록은 더 긴 길이를 요구한다', () async {
      final recorder = FakeRecorder(duration: const Duration(seconds: 4));
      final controller = makeController(recorder);

      await controller.startRecording(RecordingPurpose.enroll);
      await controller.stopAndSubmit();

      expect(controller.state.stage, VoiceStage.failure);
      expect(controller.state.localWarning, contains('5초'));
    });

    test('녹음 파일을 만들지 못하면 실패로 끝난다', () async {
      final recorder = FakeRecorder(failToProduceFile: true);
      final controller = makeController(recorder);

      await controller.startRecording(RecordingPurpose.verify);
      await controller.stopAndSubmit();

      expect(controller.state.stage, VoiceStage.failure);
    });

    test('녹음 중이 아니면 전송하지 않는다', () async {
      final controller = makeController(FakeRecorder());

      await controller.stopAndSubmit();

      expect(controller.state.stage, VoiceStage.idle);
    });
  });

  group('업로드 포맷', () {
    test('녹음 포맷이 업로드에 그대로 반영된다', () async {
      // FLAC은 무손실이라 임베딩에 영향이 없으면서 업로드가 WAV의 62%다.
      // 파일명·MIME이 실제 포맷과 어긋나면 프록시·로그를 읽기 어려워진다.
      String? uploadedName;
      adapter.onPost(
        '/api/v1/verify',
        (server) => server.reply(200, {
          'status': 'success',
          'user_id': 'alice',
          'is_verified': true,
          'match_probability': 95.4,
          'raw_cosine': 0.8,
          'scoring_method': 'raw_cosine',
          'threshold': 0.3767,
          'compared_enrollments': 1,
          'audio': {'speech_duration_sec': 3.5},
          'elapsed_ms': 116,
        }),
        data: Matchers.any,
      );
      dio.interceptors.add(InterceptorsWrapper(onRequest: (options, handler) {
        final form = options.data as FormData;
        uploadedName = form.files.first.value.filename;
        handler.next(options);
      }));

      final controller = makeController(FakeRecorder(format: UploadFormat.flac));
      await controller.startRecording(RecordingPurpose.verify);
      await controller.stopAndSubmit();

      expect(controller.state.stage, VoiceStage.success);
      expect(uploadedName, 'audio.flac');
    });

    test('WAV 폴백도 그대로 전달된다', () async {
      String? uploadedName;
      mockVerifySuccess();
      dio.interceptors.add(InterceptorsWrapper(onRequest: (options, handler) {
        final form = options.data as FormData;
        uploadedName = form.files.first.value.filename;
        handler.next(options);
      }));

      final controller = makeController(FakeRecorder(format: UploadFormat.wav));
      await controller.startRecording(RecordingPurpose.verify);
      await controller.stopAndSubmit();

      expect(uploadedName, 'audio.wav');
    });
  });

  group('취소·초기화', () {
    test('취소하면 녹음기를 멈추고 초기 상태로 돌아간다', () async {
      final recorder = FakeRecorder();
      final controller = makeController(recorder);

      await controller.startRecording(RecordingPurpose.verify);
      await controller.cancelRecording();

      expect(recorder.cancelled, isTrue);
      expect(controller.state.stage, VoiceStage.idle);
    });

    test('초기화는 결과를 지우되 목적은 유지한다', () async {
      mockVerifySuccess();
      final controller = makeController(FakeRecorder());

      await controller.startRecording(RecordingPurpose.enroll);
      controller.reset();

      expect(controller.state.stage, VoiceStage.idle);
      expect(controller.state.verifyResult, isNull);
      expect(controller.state.purpose, RecordingPurpose.enroll);
    });
  });
}
