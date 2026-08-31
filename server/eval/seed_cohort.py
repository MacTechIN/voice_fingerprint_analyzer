"""임포스터 코호트 DB 적재.

평가에 쓴 화자와 **겹치지 않는** 화자로 코호트를 만든다. 코호트에 평가·실사용
화자가 섞이면 정규화가 자기 자신을 참조하게 되어 사칭자 통계가 오염된다.

실행:
    VG_DATABASE_URL=postgresql://... .venv/bin/python -m eval.seed_cohort
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.db import session as db_session
from eval.dataset import load_utterances
from eval.extract import extract_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / ".data" / "cache"


async def main(split: str, max_per_speaker: int, replace: bool) -> None:
    settings = get_settings()
    if not settings.database_url:
        raise SystemExit("VG_DATABASE_URL이 필요합니다 (인메모리 저장소에는 적재 의미가 없음)")

    utts = load_utterances(split, max_per_speaker=max_per_speaker, seed=0)
    embeddings = extract_all(utts, CACHE_DIR / f"{split}.npz")
    logger.info("코호트 후보 %d개 (화자 %d명)", len(embeddings), len({u.speaker for u in utts}))

    await db_session.init(settings.database_url)
    repo = db_session.get_repository()
    try:
        if replace:
            removed = await repo.clear_cohort(settings.embedding_model)
            logger.info("기존 코호트 %d개 삭제", removed)

        entries = [
            (np.asarray(embeddings[u.key], dtype=np.float32).tolist(),
             settings.embedding_model,
             u.speaker)
            for u in utts
            if u.key in embeddings
        ]
        inserted = await repo.add_cohort_entries(entries, source=f"librispeech/{split}")
        logger.info("코호트 %d개 적재 완료 (model=%s)", inserted, settings.embedding_model)
    finally:
        await db_session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AS-Norm 임포스터 코호트 적재")
    parser.add_argument("--split", default="test-clean", help="LibriSpeech split")
    parser.add_argument("--max-per-speaker", type=int, default=8)
    parser.add_argument("--replace", action="store_true", help="기존 코호트를 지우고 적재")
    args = parser.parse_args()
    asyncio.run(main(args.split, args.max_per_speaker, args.replace))
