"""딥페이크 탐지의 오디오 길이별 오탐 측정 (Phase 8 후속).

**Phase 8이 배포 전 블로커로 남긴 항목이다.** ASVspoof 평가 세트는 평균 3.5초
(대부분 1구간)라, 구간별 최댓값 집계가 **긴 오디오에서 오탐을 얼마나 늘리는지**
재지 못했다. 실제로 12초 LibriSpeech 발화에서 최댓값이 0.79까지 오른 사례가
관측됐고, 클라이언트는 3~30초를 녹음한다.

최댓값 집계는 구간이 많을수록 "하나라도 높게 나올" 확률이 커진다. 진짜 사람이
말한 오디오가 길다는 이유만으로 차단되면 안 된다.

측정: **진짜 사람 음성(LibriSpeech)**을 길이별로 잘라, 길이가 늘 때 오탐률이
어떻게 변하는지 본다. 집계 방식(max/mean/trimmed)도 함께 비교한다.

실행:
    .venv/bin/python -m eval.antispoof_length_eval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from app.services import antispoof

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"
WEIGHTS = "./.model_cache/AASIST-L.pth"
RATE = 16_000

#: 클라이언트 녹음 길이 범위(3~30초)를 덮는다
LENGTHS = (4.0, 6.0, 10.0, 15.0, 20.0, 30.0)


def _chunk_scores(samples: np.ndarray) -> np.ndarray:
    """구간별 spoof 확률."""
    model = antispoof._ensure_loaded(WEIGHTS)
    chunks = antispoof._segments(samples)
    batch = torch.from_numpy(
        np.stack([np.ascontiguousarray(c, dtype=np.float32) for c in chunks])
    )
    with torch.no_grad():
        _, logits = model(batch)
        return torch.softmax(logits, dim=1)[:, 0].numpy()


def _aggregate(scores: np.ndarray) -> dict[str, float]:
    """집계 방식별 최종 점수.

    - max      : 일부만 합성인 공격을 잡지만, 구간이 많으면 오탐이 는다
    - mean     : 오탐에 강하지만 일부 합성이 희석된다
    - trimmed  : 최댓값 하나만 버린 나머지의 최댓값. 단발성 이상치는 무시하되
                 두 구간 이상이 의심스러우면 잡는다 — 둘의 절충안이다
    """
    ordered = np.sort(scores)[::-1]
    return {
        "max": float(ordered[0]),
        "mean": float(scores.mean()),
        "trimmed_max": float(ordered[1]) if len(ordered) > 1 else float(ordered[0]),
    }


def main(max_files: int, thresholds: list[float]) -> None:
    root = DATA_DIR / "LibriSpeech" / "dev-clean"
    if not root.exists():
        raise SystemExit(f"{root} 가 없습니다. eval/README.md 참조.")

    antispoof.warmup(WEIGHTS)

    # 30초를 채울 만큼 긴 발화만 고른다 — 짧은 것을 이어 붙이면 인공적인
    # 경계가 생겨 그 자체가 탐지기를 자극할 수 있다.
    candidates = []
    for path in sorted(root.rglob("*.flac")):
        info = sf.info(path)
        if info.duration >= max(LENGTHS):
            candidates.append(path)
        if len(candidates) >= max_files:
            break
    logger.info("30초 이상 발화 %d개 선정", len(candidates))
    if not candidates:
        raise SystemExit("30초 이상 발화를 찾지 못했습니다.")

    results: dict[str, dict] = {}

    for length in LENGTHS:
        need = int(RATE * length)
        per_method: dict[str, list[float]] = {"max": [], "mean": [], "trimmed_max": []}
        segment_counts = []

        for path in candidates:
            samples, sr = sf.read(path, dtype="float32", frames=need)
            if sr != RATE or len(samples) < need:
                continue
            scores = _chunk_scores(np.ascontiguousarray(samples, dtype=np.float32))
            segment_counts.append(len(scores))
            for method, value in _aggregate(scores).items():
                per_method[method].append(value)

        entry = {
            "samples": len(per_method["max"]),
            "mean_segments": float(np.mean(segment_counts)) if segment_counts else 0,
            "methods": {},
        }
        for method, values in per_method.items():
            arr = np.array(values)
            entry["methods"][method] = {
                "mean_score": float(arr.mean()),
                "max_score": float(arr.max()),
                # 진짜 사람 음성이므로 임계값을 넘으면 전부 오탐이다
                "false_alarm": {
                    str(t): float((arr >= t).mean()) for t in thresholds
                },
            }
        results[f"{length:.0f}s"] = entry
        logger.info(
            "%.0f초: 구간 %.1f개, max 평균 %.4f (오탐 @0.999 = %.1f%%)",
            length,
            entry["mean_segments"],
            entry["methods"]["max"]["mean_score"],
            entry["methods"]["max"]["false_alarm"]["0.999"] * 100,
        )

    report = {
        "model": "clovaai/aasist AASIST-L",
        "source": "LibriSpeech dev-clean (진짜 사람 음성)",
        "files": len(candidates),
        "thresholds": thresholds,
        "lengths": results,
        "note": (
            "모두 진짜 사람 음성이므로 임계값을 넘은 비율은 전부 오탐(정상 사용자 차단)이다."
        ),
    }
    path = DATA_DIR / "antispoof_length_eval.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n" + "=" * 76)
    print("딥페이크 탐지 오탐률 — 오디오 길이별 (진짜 사람 음성 LibriSpeech)")
    print(f"발화 {len(candidates)}개 · 기본 임계값 0.999 기준")
    print("-" * 76)
    print(f"{'길이':<8}{'구간':>6}{'max 오탐':>12}{'mean 오탐':>12}{'trimmed 오탐':>14}")
    print("-" * 76)
    for label, entry in results.items():
        m = entry["methods"]
        print(
            f"{label:<8}{entry['mean_segments']:>6.1f}"
            f"{m['max']['false_alarm']['0.999']*100:>11.1f}%"
            f"{m['mean']['false_alarm']['0.999']*100:>11.1f}%"
            f"{m['trimmed_max']['false_alarm']['0.999']*100:>13.1f}%"
        )
    print("-" * 76)
    print("임계값별 오탐 (max 집계):")
    header = "  " + "".join(f"{t:>10}" for t in thresholds)
    print(f"  {'길이':<6}" + header[2:])
    for label, entry in results.items():
        fa = entry["methods"]["max"]["false_alarm"]
        row = "".join(f"{fa[str(t)]*100:>9.1f}%" for t in thresholds)
        print(f"  {label:<6}{row}")
    print("=" * 76)
    print(f"\n보고서: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="딥페이크 탐지 길이별 오탐 측정")
    parser.add_argument("--max-files", type=int, default=60)
    args = parser.parse_args()
    main(args.max_files, [0.7, 0.9, 0.99, 0.999, 0.9999])
