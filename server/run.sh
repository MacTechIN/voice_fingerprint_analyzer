#!/usr/bin/env bash
# 개발 서버 실행. 프로덕션은 워커 수를 조정한다 (06 Phase 5).
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" "$@"
