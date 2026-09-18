/// 라이선스 표시 검증.
///
/// `assets/LICENSE`는 저장소 루트 파일의 **복사본**이다. Flutter는 패키지 바깥
/// 경로를 자산으로 묶지 못해 복사가 불가피한데, 복사본은 원본이 바뀌어도 조용히
/// 남는다. 낡은 라이선스가 배포되지 않도록 여기서 어긋남을 잡는다.
///
/// NOTICE는 앱에 싣지 않는다. 그 고지가 가리키는 AASIST 코드는 서버에만 있다.
/// NOTICE 내용 자체의 검증은 `server/tests/test_notice.py`가 맡는다.
library;

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:voiceguard/app_info.dart';
import 'package:voiceguard/main.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('자산 복사본이 저장소 원본과 같다', () {
    for (final name in ['LICENSE']) {
      test(name, () {
        final bundled = File('assets/$name');
        final source = File('../$name');

        expect(source.existsSync(), isTrue, reason: '저장소 루트에 $name이 없다');
        expect(bundled.existsSync(), isTrue, reason: 'assets/$name이 없다');
        expect(
          bundled.readAsStringSync(),
          source.readAsStringSync(),
          reason: '$name 복사본이 원본과 다르다. `cp ../$name assets/`로 갱신할 것',
        );
      });
    }
  });

  test('app_info의 버전이 pubspec과 같다', () {
    final pubspec = File('pubspec.yaml').readAsLinesSync();
    final line = pubspec.firstWhere((l) => l.startsWith('version:'));
    final declared = line.split(':')[1].trim();

    expect(appVersion, declared, reason: 'pubspec을 올렸으면 app_info.dart도 올릴 것');
    expect(appName, isNotEmpty);
  });

  test('registerProjectLicenses가 프로젝트 라이선스만 등록한다', () async {
    LicenseRegistry.reset();
    registerProjectLicenses();

    final entries = await LicenseRegistry.licenses.toList();
    final ours = entries
        .where((e) => e.packages.any((p) => p.startsWith('VoiceGuard')))
        .toList();

    expect(ours.length, 1);

    final texts = ours
        .map((e) => e.paragraphs.map((p) => p.text).join(' '))
        .join('\n');
    expect(texts, contains('Apache License'));

    // 서버에만 있는 구성요소의 고지가 앱에 섞여 들어가지 않아야 한다.
    expect(texts, isNot(contains('NAVER Corp')));

    LicenseRegistry.reset();
  });

  test('등록한 원문이 자산에서 실제로 읽힌다', () async {
    final license = await rootBundle.loadString('assets/LICENSE');

    expect(license, contains('Apache License'));
    expect(license, contains('Copyright 2026 MacTechIN'));
  });

  testWidgets('앱바 버튼으로 라이선스 화면을 연다', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: Builder(
        builder: (context) => Scaffold(
          appBar: AppBar(
            actions: [
              IconButton(
                key: const Key('license-button'),
                icon: const Icon(Icons.info_outline),
                tooltip: '오픈소스 라이선스',
                onPressed: () => showLicensePage(
                  context: context,
                  applicationName: appName,
                  applicationVersion: appVersion,
                ),
              ),
            ],
          ),
        ),
      ),
    ));

    expect(find.byKey(const Key('license-button')), findsOneWidget);

    await tester.tap(find.byKey(const Key('license-button')));
    await tester.pumpAndSettle();

    expect(find.byType(LicensePage), findsOneWidget);
    expect(find.text(appVersion), findsWidgets);
  });
}
