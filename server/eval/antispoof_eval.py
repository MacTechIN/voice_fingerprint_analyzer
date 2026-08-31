"""딥페이크 탐지 성능 실측 (Phase 8, FR-16).

**실제 공격 데이터로 잰다.** ASVspoof 2019 LA는 TTS·보이스 컨버전으로 만든
실제 위조 음성과 진짜 음성이 레이블과 함께 들어 있는 표준 벤치마크다. 직접
만든 합성음으로 재면 우리가 만든 공격만 잡는 모델인지 알 수 없다.

측정 항목:

* **EER** — 위조를 놓치는 비율과 진짜를 막는 비율이 같아지는 지점
* **집계 방식 비교** — 긴 오디오를 구간으로 나눌 때 max와 mean 중 무엇이 나은가
* **운영 임계값** — 진짜 사용자를 막지 않으면서 위조를 잡는 지점

실행:
    .venv/bin/python -m eval.antispoof_eval --limit 2000
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from app.services import antispoof
from eval import metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"
PARQUET = DATA_DIR / "asvspoof" / "validation.parquet"
WEIGHTS = "./.model_cache/AASIST-L.pth"

TARGET_RATE = 16_000


def _score_chunks(samples: np.ndarray) -> np.ndarray:
    """구간별 spoof 확률을 모두 반환한다 (집계 방식 비교용)."""
    import torch

    model = antispoof._ensure_loaded(WEIGHTS)
    chunks = antispoof._segments(samples)
    batch = torch.from_numpy(
        np.stack([np.ascontiguousarray(c, dtype=np.float32) for c in chunks])
    )
    with torch.no_grad():
        _, logits = model(batch)
        probs = torch.softmax(logits, dim=1)
    return probs[:, 0].numpy()


def main(limit: int, seed: int) -> None:
    if not PARQUET.exists():
        raise SystemExit(
            f"{PARQUET} 가 없습니다. eval/README.md의 ASVspoof 준비 절차를 실행하세요."
        )

    import pyarrow.parquet as pq

    logger.info("ASVspoof 2019 LA 검증 세트 로드")
    handle = pq.ParquetFile(PARQUET)
    logger.info("전체 %d개 발화", handle.metadata.num_rows)

    # 오디오 바이트가 1.5GB라 통째로 읽으면 메모리를 크게 쓴다. 먼저 레이블만
    # 읽어 뽑을 행을 정하고, 오디오는 배치 스트리밍으로 필요한 것만 가져온다.
    keys = handle.read(columns=["key"])["key"].to_pylist()
    rng = np.random.default_rng(seed)

    bona_idx = [i for i, k in enumerate(keys) if k == 0]
    spoof_idx = [i for i, k in enumerate(keys) if k == 1]
    # bonafide가 전체의 10% 정도라 무작위 추출하면 편중된다. 레이블별로 같은 수를
    # 뽑아 EER이 한쪽 분포에 치우쳐 계산되지 않게 한다.
    half = (limit // 2) if limit else min(len(bona_idx), len(spoof_idx))
    wanted = set(rng.choice(bona_idx, size=min(half, len(bona_idx)), replace=False).tolist())
    wanted |= set(rng.choice(spoof_idx, size=min(half, len(spoof_idx)), replace=False).tolist())
    logger.info("표본 %d개 (bonafide %d, spoof %d)",
                len(wanted),
                sum(1 for i in wanted if keys[i] == 0),
                sum(1 for i in wanted if keys[i] == 1))

    def _stream():
        offset = 0
        for batch in handle.iter_batches(batch_size=256):
            rows_in_batch = batch.to_pylist()
            for j, row in enumerate(rows_in_batch):
                if offset + j in wanted:
                    yield row
            offset += len(rows_in_batch)

    rows = _stream()
    total = len(wanted)

    antispoof.warmup(WEIGHTS)

    max_scores: list[float] = []
    mean_scores: list[float] = []
    labels: list[int] = []
    durations: list[float] = []
    started = time.perf_counter()
    failed = 0

    for i, row in enumerate(rows, 1):
        try:
            samples, rate = _decode(row)
        except Exception:
            failed += 1
            continue
        if rate != TARGET_RATE or len(samples) == 0:
            failed += 1
            continue

        chunk_scores = _score_chunks(samples)
        max_scores.append(float(chunk_scores.max()))
        mean_scores.append(float(chunk_scores.mean()))
        labels.append(_label(row))
        durations.append(len(samples) / rate)

        if i % 200 == 0:
            rate_per_sec = i / (time.perf_counter() - started)
            logger.info("진행 %d/%d (%.1f 발화/초)", i, total, rate_per_sec)

    y = np.array(labels)
    logger.info(
        "완료: %d개 (bonafide %d, spoof %d, 실패 %d), %.0f초",
        len(y), int((y == 0).sum()), int((y == 1).sum()), failed,
        time.perf_counter() - started,
    )

    results = {}
    for name, scores in (("max", max_scores), ("mean", mean_scores)):
        s = np.array(scores)
        # metrics.compute는 "높을수록 positive(genuine)"를 전제하므로,
        # spoof 점수는 방향이 반대다. bonafide를 positive로 두고 부호를 뒤집는다.
        m = metrics.compute(-s, 1 - y)
        results[name] = {
            "eer": m.eer,
            # 임계값도 부호를 되돌려 spoof 점수 기준으로 보고한다
            "eer_threshold": -m.eer_threshold,
            "min_dcf": m.min_dcf,
            "separation": m.separation,
            "bonafide_mean": float(s[y == 0].mean()),
            "spoof_mean": float(s[y == 1].mean()),
        }

    # 운영 관점: 진짜 사용자를 막지 않는 것이 우선이다. 후보 임계값별로
    # 위조 탐지율과 정상 차단율을 함께 본다.
    best = min(results, key=lambda k: results[k]["eer"])
    scores = np.array(max_scores if best == "max" else mean_scores)
    operating = []
    for threshold in (0.5, 0.7, 0.9, 0.95, 0.99):
        flagged = scores >= threshold
        operating.append({
            "threshold": threshold,
            "spoof_detected_ratio": float((flagged & (y == 1)).sum() / max((y == 1).sum(), 1)),
            "bonafide_blocked_ratio": float((flagged & (y == 0)).sum() / max((y == 0).sum(), 1)),
        })

    report = {
        "model": "clovaai/aasist AASIST-L",
        "dataset": "ASVspoof 2019 LA (validation)",
        "samples": len(y),
        "bonafide": int((y == 0).sum()),
        "spoof": int((y == 1).sum()),
        "mean_duration_sec": float(np.mean(durations)),
        "aggregation": results,
        "best_aggregation": best,
        "operating_points": operating,
    }
    path = DATA_DIR / "antispoof_eval.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n" + "=" * 72)
    print("딥페이크 탐지 — AASIST-L / ASVspoof 2019 LA (validation)")
    print(f"표본 {len(y)}개 (bonafide {int((y==0).sum())}, spoof {int((y==1).sum())}), "
          f"평균 {np.mean(durations):.1f}초")
    print("-" * 72)
    print(f"{'집계':<10}{'EER':>10}{'임계값':>12}{'분리도':>10}{'bonafide':>11}{'spoof':>10}")
    print("-" * 72)
    for name, r in results.items():
        print(f"{name:<10}{r['eer']*100:>9.2f}%{r['eer_threshold']:>12.4f}"
              f"{r['separation']:>10.2f}{r['bonafide_mean']:>11.4f}{r['spoof_mean']:>10.4f}")
    print("-" * 72)
    print(f"운영 임계값별 ({best} 집계):")
    print(f"  {'임계값':>8}{'위조 탐지':>12}{'정상 차단':>12}")
    for op in operating:
        print(f"  {op['threshold']:>8.2f}{op['spoof_detected_ratio']:>11.1%}"
              f"{op['bonafide_blocked_ratio']:>12.2%}")
    print("=" * 72)
    print(f"\n보고서: {path}")


def _label(row: dict) -> int:
    """1이면 spoof, 0이면 bonafide."""
    for key in ("label", "key", "attack_type", "system_id"):
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("bonafide", "bona-fide", "genuine", "real", "-"):
                return 0
            if v in ("spoof", "fake"):
                return 1
    raise KeyError(f"레이블 컬럼을 찾을 수 없습니다: {list(row)}")


def _decode(row: dict) -> tuple[np.ndarray, int]:
    """parquet 오디오 컬럼을 파형으로 디코딩한다."""
    audio = row.get("audio")
    if isinstance(audio, dict):
        if audio.get("array") is not None:
            return np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])
        if audio.get("bytes") is not None:
            data, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            return data, rate
    raise ValueError("오디오 컬럼 형식을 알 수 없습니다")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="딥페이크 탐지 성능 실측")
    parser.add_argument("--limit", type=int, default=2000, help="레이블별 균형 표본 총 개수")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.limit, args.seed)
