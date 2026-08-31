/// 서버 응답 모델.
///
/// **Tolerant Reader 원칙**: 서버는 확장 가능 스키마를 쓰며 Phase 7~8에서
/// `spoof_score` 등 필드가 최상위에 추가된다 (01 §2). 따라서 파싱은
/// **알 수 없는 필드를 무시**하고, 아직 없는 필드는 null로 받는다.
/// 필드가 늘었다고 앱이 깨지면 서버를 배포할 때마다 앱을 함께 내보내야 한다.
library;

/// VAD가 검출한 발화 구간 (초).
class SpeechSegment {
  const SpeechSegment({required this.start, required this.end});

  final double start;
  final double end;

  double get duration => end - start;

  factory SpeechSegment.fromJson(Map<String, dynamic> json) => SpeechSegment(
        start: _toDouble(json['start']) ?? 0,
        end: _toDouble(json['end']) ?? 0,
      );
}

/// 입력 오디오와 전처리 결과 요약.
///
/// 클라이언트가 녹음 품질을 스스로 진단해 사용자에게 안내하는 근거다 (04 §4).
class AudioInfo {
  const AudioInfo({
    required this.durationSec,
    required this.speechDurationSec,
    required this.speechRatio,
    required this.sampleRate,
    required this.segments,
  });

  final double durationSec;

  /// VAD 통과 유효 발화 길이. 녹음 길이와 다르다 — 무음이 빠진 값이다.
  final double speechDurationSec;

  /// 전체 대비 발화 비율. 낮으면 "더 또렷하게 말해주세요" 안내의 근거가 된다.
  final double speechRatio;

  final int sampleRate;
  final List<SpeechSegment> segments;

  factory AudioInfo.fromJson(Map<String, dynamic> json) => AudioInfo(
        durationSec: _toDouble(json['duration_sec']) ?? 0,
        speechDurationSec: _toDouble(json['speech_duration_sec']) ?? 0,
        speechRatio: _toDouble(json['speech_ratio']) ?? 0,
        sampleRate: _toInt(json['sample_rate']) ?? 0,
        segments: (json['segments'] as List<dynamic>? ?? const [])
            .whereType<Map<String, dynamic>>()
            .map(SpeechSegment.fromJson)
            .toList(),
      );
}

/// 등록 결과.
class EnrollResult {
  const EnrollResult({
    required this.userId,
    required this.enrollmentId,
    required this.replaced,
    required this.audio,
    required this.elapsedMs,
  });

  final String userId;
  final int enrollmentId;

  /// 이번 등록으로 비활성화된 기존 성문 수. 0이면 첫 등록이다.
  final int replaced;

  final AudioInfo audio;
  final double elapsedMs;

  bool get isReEnrollment => replaced > 0;

  factory EnrollResult.fromJson(Map<String, dynamic> json) => EnrollResult(
        userId: json['user_id'] as String? ?? '',
        enrollmentId: _toInt(json['enrollment_id']) ?? 0,
        replaced: _toInt(json['replaced']) ?? 0,
        audio: AudioInfo.fromJson(
            json['audio'] as Map<String, dynamic>? ?? const {}),
        elapsedMs: _toDouble(json['elapsed_ms']) ?? 0,
      );
}

/// 검증 결과.
class VerifyResult {
  const VerifyResult({
    required this.userId,
    required this.isVerified,
    required this.matchProbability,
    required this.rawCosine,
    required this.threshold,
    required this.scoringMethod,
    required this.comparedEnrollments,
    required this.audio,
    required this.elapsedMs,
    this.normalizedScore,
  });

  final String userId;
  final bool isVerified;

  /// 0~100 척도의 일치도.
  ///
  /// **확률이 아니다.** 판정 경계를 50%에 맞춘 구간 선형 재척도이므로,
  /// UI에서 "확률"이라고 표기하지 않는다.
  final double matchProbability;

  final double rawCosine;

  /// AS-Norm 정규화 점수. 서버에 코호트가 없으면 null이다.
  final double? normalizedScore;

  final double threshold;

  /// 판정 근거 점수 종류 (`as_norm` | `raw_cosine`).
  final String scoringMethod;

  final int comparedEnrollments;
  final AudioInfo audio;
  final double elapsedMs;

  /// 정규화 없이 판정됐는지. 서버 운영 상태를 UI에서 드러낼 근거다.
  bool get isFallbackScoring => scoringMethod != 'as_norm';

  factory VerifyResult.fromJson(Map<String, dynamic> json) => VerifyResult(
        userId: json['user_id'] as String? ?? '',
        isVerified: json['is_verified'] as bool? ?? false,
        matchProbability: _toDouble(json['match_probability']) ?? 0,
        rawCosine: _toDouble(json['raw_cosine']) ?? 0,
        normalizedScore: _toDouble(json['normalized_score']),
        threshold: _toDouble(json['threshold']) ?? 0,
        scoringMethod: json['scoring_method'] as String? ?? 'raw_cosine',
        comparedEnrollments: _toInt(json['compared_enrollments']) ?? 0,
        audio: AudioInfo.fromJson(
            json['audio'] as Map<String, dynamic>? ?? const {}),
        elapsedMs: _toDouble(json['elapsed_ms']) ?? 0,
      );
}

/// 서버 상태.
class ServerHealth {
  const ServerHealth({
    required this.status,
    required this.model,
    required this.modelsLoaded,
    required this.storage,
    required this.storageOk,
    required this.asnormActive,
    required this.cohortSize,
    required this.embeddingBackend,
  });

  final String status;
  final String model;
  final bool modelsLoaded;
  final String storage;
  final bool storageOk;

  /// AS-Norm 정규화가 실제로 적용되는지. false면 판정 신뢰도가 낮다.
  final bool asnormActive;

  final int cohortSize;
  final String embeddingBackend;

  bool get isReady => status == 'ok' && modelsLoaded && storageOk;

  factory ServerHealth.fromJson(Map<String, dynamic> json) => ServerHealth(
        status: json['status'] as String? ?? 'unknown',
        model: json['model'] as String? ?? '',
        modelsLoaded: json['models_loaded'] as bool? ?? false,
        storage: json['storage'] as String? ?? 'unknown',
        storageOk: json['storage_ok'] as bool? ?? false,
        asnormActive: json['asnorm_active'] as bool? ?? false,
        cohortSize: _toInt(json['cohort_size']) ?? 0,
        embeddingBackend: json['embedding_backend'] as String? ?? 'unknown',
      );
}

// --- JSON 숫자 파싱 ---
//
// 서버가 정수로 보낸 값(예: elapsed_ms가 딱 떨어질 때)이 Dart에서 int로 오므로
// `as double`은 런타임에 터진다. 숫자는 반드시 num을 거쳐 변환한다.

double? _toDouble(Object? value) =>
    value is num ? value.toDouble() : (value is String ? double.tryParse(value) : null);

int? _toInt(Object? value) =>
    value is num ? value.toInt() : (value is String ? int.tryParse(value) : null);
