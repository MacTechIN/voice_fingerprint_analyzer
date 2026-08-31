/// 서버 반려 사유와 사용자 안내 문구.
///
/// 서버는 발화 부족·무음 등을 422 + `code`로 반려한다. 이 코드값은 서버가
/// **변경 금지 계약**으로 관리하며, 클라이언트는 사유별로 다른 재녹음 안내를
/// 띄운다 (04 §4).
library;

/// 서버가 정의한 반려 사유.
enum RejectionCode {
  emptyFile('empty_file'),
  fileTooLarge('file_too_large'),
  unreadableAudio('unreadable_audio'),
  audioTooLong('audio_too_long'),
  noSpeechDetected('no_speech_detected'),
  speechTooShort('speech_too_short'),
  notEnrolled('not_enrolled'),
  modelMismatch('model_mismatch'),

  /// 서버가 새 사유 코드를 추가했는데 앱이 아직 모르는 경우.
  ///
  /// 이 값이 있어야 앱이 서버 배포에 끌려다니지 않는다. 모르는 코드는
  /// 서버가 준 `detail` 문구를 그대로 보여주면 된다.
  unknown('unknown');

  const RejectionCode(this.wireValue);

  final String wireValue;

  static RejectionCode fromWire(String? value) {
    for (final code in RejectionCode.values) {
      if (code.wireValue == value) return code;
    }
    return RejectionCode.unknown;
  }
}

/// API 호출 실패.
class ApiException implements Exception {
  const ApiException({
    required this.message,
    this.code = RejectionCode.unknown,
    this.statusCode,
    this.isNetworkError = false,
  });

  /// 사용자에게 보여줄 메시지.
  final String message;

  final RejectionCode code;
  final int? statusCode;

  /// 네트워크 문제(연결 실패·타임아웃)인지. 재시도 안내를 다르게 해야 한다.
  final bool isNetworkError;

  /// 사용자가 다시 녹음하면 해결될 수 있는 문제인지.
  bool get isRetryableByReRecording => switch (code) {
        RejectionCode.emptyFile ||
        RejectionCode.unreadableAudio ||
        RejectionCode.noSpeechDetected ||
        RejectionCode.speechTooShort ||
        RejectionCode.audioTooLong ||
        RejectionCode.fileTooLarge =>
          true,
        _ => false,
      };

  /// 사유별 행동 안내.
  ///
  /// 서버의 `detail`은 무엇이 잘못됐는지 알려주고, 이 문구는 **무엇을 하면
  /// 되는지** 알려준다. 둘을 함께 보여준다.
  String get actionHint => switch (code) {
        RejectionCode.noSpeechDetected =>
          '조용한 곳에서 마이크에 가까이 대고 또렷하게 말해주세요.',
        RejectionCode.speechTooShort =>
          '조금 더 길게 말해주세요. 문장 하나를 천천히 읽으면 충분합니다.',
        RejectionCode.unreadableAudio =>
          '녹음이 손상됐습니다. 다시 녹음해주세요.',
        RejectionCode.audioTooLong || RejectionCode.fileTooLarge =>
          '녹음이 너무 깁니다. 짧게 다시 녹음해주세요.',
        RejectionCode.emptyFile => '녹음된 소리가 없습니다. 다시 시도해주세요.',
        RejectionCode.notEnrolled => '먼저 음성을 등록해주세요.',
        RejectionCode.modelMismatch =>
          '분석 모델이 갱신되어 기존 등록을 쓸 수 없습니다. 다시 등록해주세요.',
        RejectionCode.unknown => '',
      };

  @override
  String toString() => 'ApiException($code): $message';
}
