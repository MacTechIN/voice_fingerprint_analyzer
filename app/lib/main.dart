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

/// 프로젝트 라이선스와 서드파티 고지를 Flutter 라이선스 화면에 등록한다.
///
/// Flutter는 의존 패키지의 LICENSE를 자동으로 모아 보여주지만, 이 저장소 자체의
/// 라이선스와 NOTICE는 모르므로 직접 넣어준다. NOTICE는 재배포 시 함께 전달해야
/// 하는 고지라 앱에서도 확인할 수 있어야 한다.
void registerProjectLicenses() {
  LicenseRegistry.addLicense(() async* {
    yield LicenseEntryWithLineBreaks(
      const ['VoiceGuard'],
      await rootBundle.loadString('assets/LICENSE'),
    );
    yield LicenseEntryWithLineBreaks(
      const ['VoiceGuard 서드파티 고지 (NOTICE)'],
      await rootBundle.loadString('assets/NOTICE'),
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
