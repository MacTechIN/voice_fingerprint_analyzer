/// VoiceGuard 클라이언트 진입점.
///
/// 앱은 UI와 녹음만 담당하고 분석은 전부 서버가 한다 (01 §2 Thin Client).
/// 따라서 여기에는 딥러닝 라이브러리도, 모델 초기화도 없다.
library;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'ui/screens/voice_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  registerProjectLicenses();
  runApp(const ProviderScope(child: VoiceGuardApp()));
}

/// 이 저장소의 라이선스를 Flutter 라이선스 화면에 등록한다.
///
/// Flutter는 의존 패키지의 LICENSE를 자동으로 모아 보여주지만 이 저장소 자체의
/// 라이선스는 모르므로 직접 넣어준다.
///
/// **NOTICE는 싣지 않는다.** NOTICE가 담은 고지는 서버가 포함한 AASIST(MIT)에
/// 대한 것인데, 그 코드는 앱 바이너리에 들어가지 않는다. 앱이 재배포하지 않는
/// 구성요소의 고지를 앱에 실으면 사용자가 앱에 포함된 것으로 오해한다.
/// 저장소 재배포 시의 고지 의무는 루트 NOTICE 파일이 그대로 진다.
void registerProjectLicenses() {
  LicenseRegistry.addLicense(() async* {
    yield LicenseEntryWithLineBreaks(
      const ['VoiceGuard'],
      await rootBundle.loadString('assets/LICENSE'),
    );
  });
}

class VoiceGuardApp extends StatelessWidget {
  const VoiceGuardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'VoiceGuard',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const VoiceScreen(),
    );
  }
}
