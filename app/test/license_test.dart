/// 라이선스 표시 검증.
///
/// `assets/`의 LICENSE·NOTICE는 저장소 루트 파일의 **복사본**이다. Flutter는
/// 패키지 바깥 경로를 자산으로 묶지 못해 복사가 불가피한데, 복사본은 원본이
/// 바뀌어도 조용히 남는다. 라이선스 고지가 낡은 채로 배포되는 것은 단순한
/// 문서 불일치가 아니라 재배포 조건 위반이 될 수 있어, 여기서 어긋남을 잡는다.
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
    for (final name in ['LICENSE', 'NOTICE']) {
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

  test('NOTICE에 벤더링된 AASIST의 MIT 문구가 들어 있다', () {
    final notice = File('assets/NOTICE').readAsStringSync();

    // 헤더의 "MIT license" 한 줄만으로는 MIT 조건을 만족하지 못한다.
    // 저작권 표시와 허가 문구가 함께 있어야 한다.
    expect(notice, contains('Copyright (c) 2021-present NAVER Corp.'));
    expect(notice, contains('Permission is hereby granted'));
    expect(notice, contains('THE SOFTWARE IS PROVIDED "AS IS"'));
  });

  test('app_info의 버전이 pubspec과 같다', () {
    final pubspec = File('pubspec.yaml').readAsLinesSync();
    final line = pubspec.firstWhere((l) => l.startsWith('version:'));
    final declared = line.split(':')[1].trim();

    expect(appVersion, declared, reason: 'pubspec을 올렸으면 app_info.dart도 올릴 것');
    expect(appName, isNotEmpty);
  });

  test('registerProjectLicenses가 두 원문을 등록한다', () async {
    LicenseRegistry.reset();
    registerProjectLicenses();

    final entries = await LicenseRegistry.licenses.toList();
    final ours = entries
        .where((e) => e.packages.any((p) => p.startsWith('VoiceGuard')))
        .toList();

    expect(ours.length, 2);

    final texts = ours
        .map((e) => e.paragraphs.map((p) => p.text).join(' '))
        .join('\n');
    expect(texts, contains('Apache License'));
    expect(texts, contains('NAVER Corp'));

    LicenseRegistry.reset();
  });

  test('등록한 원문이 자산에서 실제로 읽힌다', () async {
    final license = await rootBundle.loadString('assets/LICENSE');
    final notice = await rootBundle.loadString('assets/NOTICE');

    expect(license, contains('Apache License'));
    expect(license, contains('Copyright 2026 MacTechIN'));
    expect(notice, contains('THIRD-PARTY SOFTWARE'));
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
