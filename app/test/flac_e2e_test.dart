/// FLAC/WAV 업로드가 동일 결과를 내는지 실서버로 확인.
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:voiceguard/api/client.dart';
import 'package:voiceguard/audio/recording_policy.dart';

const _base = String.fromEnvironment('VG_E2E_BASE_URL');
const _dir = String.fromEnvironment('VG_E2E_AUDIO_DIR');

void main() {
  if (_base.isEmpty) { test('skip', () {}, skip: true); return; }
  final api = VoiceGuardApi(baseUrl: _base);
  final user = 'flac-${DateTime.now().millisecondsSinceEpoch}';

  test('FLAC과 WAV가 동일한 성문 점수를 낸다', () async {
    await api.enroll(
      userId: user,
      audio: File('$_dir/spk0_0.wav'),
      format: UploadFormat.wav,
    );

    final wav = await api.verify(
      userId: user, audio: File('$_dir/spk0_1.wav'), format: UploadFormat.wav);
    final flac = await api.verify(
      userId: user, audio: File('$_dir/spk0_1.flac'), format: UploadFormat.flac);

    // 무손실이므로 임베딩이 비트 단위로 같아야 한다
    expect(flac.rawCosine, closeTo(wav.rawCosine, 1e-6));
    expect(flac.isVerified, wav.isVerified);

    final wavSize = File('$_dir/spk0_1.wav').lengthSync();
    final flacSize = File('$_dir/spk0_1.flac').lengthSync();
    // ignore: avoid_print
    print('  WAV ${(wavSize/1024).round()}KB → FLAC ${(flacSize/1024).round()}KB '
          '(${(flacSize/wavSize*100).round()}%), cosine ${wav.rawCosine} == ${flac.rawCosine}');
    expect(flacSize, lessThan(wavSize));
  });
}
