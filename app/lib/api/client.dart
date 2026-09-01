/// VoiceGuard API 클라이언트.
///
/// 앱은 딥러닝 추론을 하지 않는다 (01 §2 Thin Client). 오디오를 서버로 보내고
/// 결과를 받는 것이 전부이므로, 이 계층이 앱의 유일한 분석 경로다.
library;

import 'dart:io';

import 'package:dio/dio.dart';

import '../audio/recording_policy.dart';
import 'errors.dart';
import 'models.dart';

/// 기본 API 베이스 URL.
///
/// 실행 시 `--dart-define=VG_API_BASE_URL=...`로 덮어쓴다. 안드로이드 에뮬레이터에서
/// 호스트를 가리키려면 `http://10.0.2.2:8000`을 쓴다.
const String kDefaultApiBaseUrl = String.fromEnvironment(
  'VG_API_BASE_URL',
  defaultValue: 'http://localhost:8000',
);

class VoiceGuardApi {
  VoiceGuardApi({Dio? dio, String baseUrl = kDefaultApiBaseUrl})
      : _dio = dio ??
            Dio(BaseOptions(
              baseUrl: baseUrl,
              // 서버는 VAD → 임베딩 → AS-Norm을 순차 처리한다. 실측 검증 지연이
              // 약 120ms지만, 콜드 스타트와 네트워크 지연을 감안해 여유를 둔다.
              connectTimeout: const Duration(seconds: 10),
              sendTimeout: const Duration(seconds: 30),
              receiveTimeout: const Duration(seconds: 30),
              // 4xx를 예외로 만들지 않고 직접 처리한다. 422는 서버 장애가 아니라
              // 입력 문제이며, 본문의 code를 읽어야 안내를 분기할 수 있다.
              validateStatus: (status) => status != null && status < 500,
            ));

  final Dio _dio;

  static const String _prefix = '/api/v1';

  /// 서버 상태 조회.
  Future<ServerHealth> health() async {
    final response = await _send(() => _dio.get<Map<String, dynamic>>('$_prefix/health'));
    return ServerHealth.fromJson(response);
  }

  /// 성문 등록.
  Future<EnrollResult> enroll({
    required String userId,
    required File audio,
    UploadFormat format = UploadFormat.wav,
  }) async {
    final response = await _send(
      () => _dio.post<Map<String, dynamic>>(
        '$_prefix/enroll',
        data: _multipart(userId, audio, format),
      ),
    );
    return EnrollResult.fromJson(response);
  }

  /// 성문 검증.
  Future<VerifyResult> verify({
    required String userId,
    required File audio,
    UploadFormat format = UploadFormat.wav,
  }) async {
    final response = await _send(
      () => _dio.post<Map<String, dynamic>>(
        '$_prefix/verify',
        data: _multipart(userId, audio, format),
      ),
    );
    return VerifyResult.fromJson(response);
  }

  /// 업로드 본문 구성.
  ///
  /// 서버는 파일 내용으로 포맷을 판별하므로 파일명·MIME이 틀려도 동작하지만,
  /// 맞춰 보내는 편이 프록시·로그를 읽기 쉽다.
  FormData _multipart(String userId, File audio, UploadFormat format) {
    return FormData.fromMap({
      'user_id': userId,
      'file': MultipartFile.fromFileSync(
        audio.path,
        filename: format.fileName,
        contentType: DioMediaType.parse(format.mimeType),
      ),
    });
  }

  /// 요청 실행과 오류 변환.
  ///
  /// 서버의 422 반려를 [ApiException]으로 바꿔, 호출부가 네트워크 오류와
  /// 입력 반려를 같은 방식으로 다루게 한다.
  Future<Map<String, dynamic>> _send(
    Future<Response<Map<String, dynamic>>> Function() request,
  ) async {
    final Response<Map<String, dynamic>> response;
    try {
      response = await request();
    } on DioException catch (e) {
      throw ApiException(
        message: _networkMessage(e),
        isNetworkError: true,
      );
    }

    final body = response.data ?? const <String, dynamic>{};
    final status = response.statusCode ?? 0;

    if (status >= 200 && status < 300) {
      return body;
    }

    // 서버가 준 사유 코드와 설명을 그대로 살린다. 앱이 모르는 코드여도
    // detail을 보여주면 사용자는 무엇이 문제인지 알 수 있다.
    final code = RejectionCode.fromWire(body['code'] as String?);
    final detail = body['detail'] as String?;
    throw ApiException(
      message: detail?.isNotEmpty == true ? detail! : '요청을 처리하지 못했습니다. (HTTP $status)',
      code: code,
      statusCode: status,
    );
  }

  String _networkMessage(DioException e) => switch (e.type) {
        DioExceptionType.connectionTimeout ||
        DioExceptionType.sendTimeout ||
        DioExceptionType.receiveTimeout =>
          '서버 응답이 지연되고 있습니다. 잠시 후 다시 시도해주세요.',
        DioExceptionType.connectionError =>
          '서버에 연결할 수 없습니다. 네트워크 상태를 확인해주세요.',
        DioExceptionType.cancel => '요청이 취소되었습니다.',
        _ => '네트워크 오류가 발생했습니다. 다시 시도해주세요.',
      };
}
