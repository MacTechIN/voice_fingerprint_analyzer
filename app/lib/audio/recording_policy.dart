/// 녹음 품질 정책.
///
/// 서버 임베딩 모델은 짧거나 오염된 발화에서 신뢰도가 급락하므로, 클라이언트가
/// UI 수준에서 이를 강제·유도한다 (04 §4). 서버에 보내기 전에 걸러내면 왕복
/// 시간과 통신 비용을 아끼고, 사용자는 더 빨리 피드백을 받는다.
///
/// 이 파일에 정책만 모아둔 이유는 서버 설정(`VG_MIN_SPEECH_SEC` 등)과 짝을
/// 이루기 때문이다. 서버 기준이 바뀌면 여기도 함께 봐야 한다.
library;

/// 녹음 규격 — 서버 처리 규격(02 §1)과 일치시킨다.
class RecordingSpec {
  const RecordingSpec._();

  /// 서버 내부 처리 샘플레이트. 클라이언트가 맞춰 보내면 서버 리샘플링 왜곡이 없다.
  static const int sampleRate = 16000;

  /// 모노. 스테레오를 보내면 서버가 평균 내어 합치므로 대역만 낭비된다.
  static const int channels = 1;

  /// 검증 최소 녹음 길이.
  ///
  /// 서버의 유효 발화 하한은 1.5초지만, VAD가 무음을 걷어내면 녹음 길이보다
  /// 짧아지므로 여유를 둔다.
  static const Duration minVerifyDuration = Duration(seconds: 3);

  /// 등록 권장 최소 길이.
  ///
  /// 등록 성문은 이후 모든 검증의 기준이 되므로 검증보다 길게 받는다.
  static const Duration minEnrollDuration = Duration(seconds: 5);

  /// 최대 녹음 길이. 서버 허용치(300초)보다 훨씬 짧게 잡아 사용자가 무한정
  /// 녹음하지 않게 한다.
  static const Duration maxDuration = Duration(seconds: 30);

  /// 입력 레벨 경고 기준 (dBFS).
  ///
  /// record 패키지의 진폭은 dBFS로 오며 0이 최대다.
  static const double tooQuietDbfs = -45;
  static const double clippingDbfs = -1.5;
}

/// 녹음 목적.
enum RecordingPurpose {
  enroll,
  verify;

  Duration get minDuration => switch (this) {
        RecordingPurpose.enroll => RecordingSpec.minEnrollDuration,
        RecordingPurpose.verify => RecordingSpec.minVerifyDuration,
      };

  String get guidance => switch (this) {
        RecordingPurpose.enroll =>
          '조용한 곳에서 아래 문장을 또렷하게 읽어주세요. 5초 이상 녹음됩니다.',
        RecordingPurpose.verify => '아래 문장을 또렷하게 읽어주세요. 3초 이상 필요합니다.',
      };
}

/// 발화 유도 스크립트.
///
/// 자유 발화보다 정해진 문장을 읽게 하면 음소 다양성이 확보되고 길이도
/// 일정해진다 (04 §4).
class PromptScripts {
  const PromptScripts._();

  static const List<String> enroll = [
    '오늘 날씨가 참 좋아서 공원에 산책을 다녀왔습니다.',
    '기술은 사람의 삶을 편리하게 만드는 도구입니다.',
    '작은 습관이 모여 큰 변화를 만들어 냅니다.',
  ];

  static const List<String> verify = [
    '제 목소리로 본인 인증을 진행합니다.',
    '오늘 하루도 좋은 일만 가득하기를 바랍니다.',
  ];

  /// 목적에 맞는 문장을 순환 선택한다.
  static String pick(RecordingPurpose purpose, int index) {
    final list = purpose == RecordingPurpose.enroll ? enroll : verify;
    return list[index.abs() % list.length];
  }
}

/// 입력 레벨 상태 — 사용자에게 실시간 경고를 띄우는 근거.
enum InputLevelStatus {
  tooQuiet,
  good,
  clipping;

  static InputLevelStatus fromDbfs(double dbfs) {
    if (dbfs >= RecordingSpec.clippingDbfs) return InputLevelStatus.clipping;
    if (dbfs <= RecordingSpec.tooQuietDbfs) return InputLevelStatus.tooQuiet;
    return InputLevelStatus.good;
  }

  String? get warning => switch (this) {
        InputLevelStatus.tooQuiet => '소리가 너무 작습니다. 마이크에 가까이 대고 말해주세요.',
        InputLevelStatus.clipping => '소리가 너무 큽니다. 마이크에서 조금 떨어져 주세요.',
        InputLevelStatus.good => null,
      };
}

/// 녹음 길이 검사 결과.
class DurationCheck {
  const DurationCheck({required this.isAcceptable, this.message});

  final bool isAcceptable;
  final String? message;

  /// 서버로 보내기 전에 길이를 검사한다.
  ///
  /// 짧은 녹음을 그대로 보내면 서버가 422로 반려하는데, 왕복 시간을 쓰고
  /// 나서야 "더 길게 말하세요"를 듣게 된다. 앱에서 먼저 막는 편이 낫다.
  factory DurationCheck.evaluate(Duration recorded, RecordingPurpose purpose) {
    final minimum = purpose.minDuration;
    if (recorded < minimum) {
      final seconds = minimum.inSeconds;
      return DurationCheck(
        isAcceptable: false,
        message: '$seconds초 이상 녹음해주세요. (현재 ${recorded.inSeconds}초)',
      );
    }
    if (recorded > RecordingSpec.maxDuration) {
      return DurationCheck(
        isAcceptable: false,
        message: '${RecordingSpec.maxDuration.inSeconds}초 이내로 녹음해주세요.',
      );
    }
    return const DurationCheck(isAcceptable: true);
  }
}
