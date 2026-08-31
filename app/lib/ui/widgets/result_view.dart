/// 결과 시각화 (FR-07).
library;

import 'package:flutter/material.dart';

import '../../api/errors.dart';
import '../../api/models.dart';

/// 검증 결과 — 일치도를 원형 게이지로 보여준다.
class VerifyResultView extends StatelessWidget {
  const VerifyResultView({super.key, required this.result});

  final VerifyResult result;

  @override
  Widget build(BuildContext context) {
    final verified = result.isVerified;
    final color = verified ? Colors.green : Colors.red;

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        // 애니메이션 게이지 — 값이 차오르며 결과를 전달한다
        TweenAnimationBuilder<double>(
          tween: Tween(begin: 0, end: result.matchProbability / 100),
          duration: const Duration(milliseconds: 900),
          curve: Curves.easeOutCubic,
          builder: (context, value, _) => SizedBox(
            width: 180,
            height: 180,
            child: Stack(
              alignment: Alignment.center,
              children: [
                SizedBox(
                  width: 180,
                  height: 180,
                  child: CircularProgressIndicator(
                    value: value,
                    strokeWidth: 14,
                    backgroundColor: Colors.grey.shade200,
                    valueColor: AlwaysStoppedAnimation<Color>(color),
                  ),
                ),
                Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      '${(value * 100).toStringAsFixed(1)}%',
                      key: const Key('match-probability'),
                      style: Theme.of(context)
                          .textTheme
                          .headlineMedium
                          ?.copyWith(fontWeight: FontWeight.bold, color: color),
                    ),
                    // "확률"이라고 쓰지 않는다 — 판정 경계를 50%에 맞춘 재척도이지
                    // 확률이 아니다 (server/README).
                    Text('일치도', style: Theme.of(context).textTheme.bodySmall),
                  ],
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 20),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(verified ? Icons.verified_user : Icons.gpp_bad, color: color),
            const SizedBox(width: 8),
            Text(
              verified ? '본인 확인됨' : '본인 확인 실패',
              key: const Key('verify-verdict'),
              style: Theme.of(context)
                  .textTheme
                  .titleLarge
                  ?.copyWith(color: color, fontWeight: FontWeight.bold),
            ),
          ],
        ),
        const SizedBox(height: 16),
        _Details(result: result),
      ],
    );
  }
}

class _Details extends StatelessWidget {
  const _Details({required this.result});

  final VerifyResult result;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          children: [
            _row('유효 발화', '${result.audio.speechDurationSec.toStringAsFixed(1)}초'),
            _row('대조한 등록', '${result.comparedEnrollments}건'),
            _row('응답 시간', '${result.elapsedMs.toStringAsFixed(0)}ms'),
            if (result.isFallbackScoring)
              // 정규화가 꺼진 채 판정된 사실을 숨기지 않는다. 서버 운영 문제이며,
              // 이 상태의 판정은 신뢰도가 낮다 (server/README).
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Row(
                  key: const Key('fallback-warning'),
                  children: [
                    Icon(Icons.info_outline,
                        size: 16, color: Colors.orange.shade800),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        '점수 정규화 없이 판정되었습니다 (서버 코호트 미적재).',
                        style: TextStyle(
                            fontSize: 12, color: Colors.orange.shade800),
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _row(String label, String value) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 3),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: const TextStyle(color: Colors.black54)),
            Text(value, style: const TextStyle(fontWeight: FontWeight.w600)),
          ],
        ),
      );
}

/// 등록 결과.
class EnrollResultView extends StatelessWidget {
  const EnrollResultView({super.key, required this.result});

  final EnrollResult result;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(Icons.check_circle, size: 96, color: Colors.green.shade600),
        const SizedBox(height: 16),
        Text(
          result.isReEnrollment ? '음성이 다시 등록되었습니다' : '음성이 등록되었습니다',
          key: const Key('enroll-verdict'),
          style: Theme.of(context).textTheme.titleLarge,
        ),
        const SizedBox(height: 8),
        Text(
          '유효 발화 ${result.audio.speechDurationSec.toStringAsFixed(1)}초',
          style: Theme.of(context).textTheme.bodyMedium,
        ),
        if (result.isReEnrollment)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Text(
              '기존 등록 ${result.replaced}건은 더 이상 사용되지 않습니다.',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
            ),
          ),
      ],
    );
  }
}

/// 실패 안내.
///
/// 서버의 `detail`(무엇이 잘못됐는지)과 앱의 `actionHint`(무엇을 하면 되는지)를
/// 함께 보여준다.
class FailureView extends StatelessWidget {
  const FailureView({super.key, this.error, this.localWarning});

  final ApiException? error;
  final String? localWarning;

  @override
  Widget build(BuildContext context) {
    final message = localWarning ?? error?.message ?? '알 수 없는 오류가 발생했습니다.';
    final hint = localWarning != null ? null : error?.actionHint;

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(Icons.error_outline, size: 72, color: Colors.orange.shade700),
        const SizedBox(height: 16),
        Text(
          message,
          key: const Key('failure-message'),
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.titleMedium,
        ),
        if (hint != null && hint.isNotEmpty) ...[
          const SizedBox(height: 8),
          Text(
            hint,
            key: const Key('failure-hint'),
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.grey.shade700),
          ),
        ],
      ],
    );
  }
}
