/// VoiceGuard 클라이언트 진입점.
///
/// 앱은 UI와 녹음만 담당하고 분석은 전부 서버가 한다 (01 §2 Thin Client).
/// 따라서 여기에는 딥러닝 라이브러리도, 모델 초기화도 없다.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'ui/screens/voice_screen.dart';

void main() {
  runApp(const ProviderScope(child: VoiceGuardApp()));
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
