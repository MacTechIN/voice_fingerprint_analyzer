"""NOTICE 파일의 서드파티 고지 검증.

`server/vendor/`에는 외부 저장소에서 그대로 가져온 소스가 들어 있다. MIT는
저작권 표시와 허가 문구를 사본에 포함할 것을 조건으로 걸므로, 그 문구가 빠지면
저장소를 재배포할 때 조건을 충족하지 못한다. 파일 헤더의 "MIT license" 한 줄은
허가 문구가 아니다.

이 검사가 앱이 아니라 서버 쪽에 있는 이유는 벤더링된 코드가 여기에만 있기
때문이다. 앱 바이너리에는 들어가지 않아 앱 라이선스 화면에는 싣지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = REPO_ROOT / "server" / "vendor"

#: MIT 허가 문구의 핵심 구절. 하나라도 빠지면 조건을 만족하지 못한다.
MIT_REQUIRED = (
    "Permission is hereby granted",
    "The above copyright notice and this permission notice shall be included",
    'THE SOFTWARE IS PROVIDED "AS IS"',
)


@pytest.fixture(scope="module")
def notice() -> str:
    path = REPO_ROOT / "NOTICE"
    assert path.exists(), "저장소 루트에 NOTICE가 없다"
    return path.read_text(encoding="utf-8")


def test_notice_declares_project_copyright(notice: str) -> None:
    """프로젝트 자체 저작권과 적용 라이선스를 밝힌다."""
    assert "Copyright 2026 MacTechIN" in notice
    assert "Apache License" in notice


def test_notice_carries_aasist_mit_text(notice: str) -> None:
    """벤더링한 AASIST의 저작권 표시와 MIT 허가 문구 전문을 담는다."""
    assert "Copyright (c) 2021-present NAVER Corp." in notice
    for clause in MIT_REQUIRED:
        assert clause in notice, f"MIT 허가 문구 누락: {clause!r}"


def test_every_vendored_file_is_listed(notice: str) -> None:
    """벤더링된 파일이 빠짐없이 NOTICE에 적혀 있다.

    새 서드파티 파일을 vendor/에 넣고 고지를 잊는 것이 가장 흔한 실수다.
    파일을 추가하면 이 테스트가 먼저 실패하도록 한다.
    """
    vendored = sorted(
        p for p in VENDOR_DIR.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    assert vendored, "vendor/ 가 비어 있다 — 경로가 바뀌었는지 확인할 것"

    missing = [
        p.relative_to(REPO_ROOT).as_posix()
        for p in vendored
        if p.relative_to(REPO_ROOT).as_posix() not in notice
    ]
    assert not missing, f"NOTICE에 적히지 않은 벤더링 파일: {missing}"


def test_vendored_source_keeps_original_header() -> None:
    """원저작권 헤더를 지우지 않았다.

    NOTICE에 적는 것과 별개로, 원본 파일 안의 표시도 그대로 두어야 한다.
    """
    header = (VENDOR_DIR / "aasist_model.py").read_text(encoding="utf-8")[:400]
    assert "Copyright (c) 2021-present NAVER Corp." in header
    assert "MIT license" in header
