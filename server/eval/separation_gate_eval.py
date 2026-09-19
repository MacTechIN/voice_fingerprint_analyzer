"""분리 게이트 임계값 캘리브레이션 (Phase 7 후속).

**문제.** `VG_SEPARATION_ENABLED`는 켜고 끄는 것뿐이다. 다중 화자가 "섞여 들어올
수도 있는" 배포에서는 켜야 하는데, 그러면 단일 화자 요청까지 전부 분리를 거친다.
분리는 6초 오디오에 약 7초가 들고, 단일 화자에 걸면 아티팩트가 임베딩을 훼손한다
(Phase 7 실측: 깨끗한 오디오 EER 0.83% → 분리 후 2.50%). 즉 **켜는 순간 단일 화자
사용자는 느려지고 동시에 정확도까지 잃는다.**

**게이트.** 원본을 그대로 임베딩해 등록 성문과 대조하고, 점수가 임계값 이상이면
분리를 건너뛴다. 분리의 목적은 혼합에서 타겟을 되찾는 것인데, 원본이 이미 등록
화자와 충분히 닮았다면 되찾을 것이 없기 때문이다.

임계값을 낮추면 분리를 덜 호출해 싸고 깨끗해지지만, 진짜 혼합까지 건너뛰어
놓칠 수 있다. 이 도구는 임계값을 훑어 **혼합 정확도를 잃지 않는 가장 싼 지점**을
찾는다.

측정 격자 — 입력 종류 × 게이트 임계값:

    입력      | 게이트 없음(항상 분리) | 임계값 t | 분리 안 함
    ----------|------------------------|----------|------------
    깨끗       |           ✓            |    ✓     |     ✓
    2인 혼합    |           ✓            |    ✓     |     ✓

**근사 하나를 쓴다.** 분리 출력 선택에는 등록 성문이 필요한데, 모든 (검증, 등록)
쌍마다 분리를 돌리면 트라이얼 수의 제곱이 되어 비현실적이다. 그래서 분리는 트라이얼당
한 번, 진짜 등록 성문으로 선택해 두고 impostor 점수는 그 결과를 재사용한다.
실제 서비스는 사칭당한 쪽의 성문으로 선택하므로 다른(대개 더 낮은) 점수가 나온다 —
**이 근사는 impostor에게 유리한 쪽이라 측정된 EER은 보수적이다.**

게이트는 원시 코사인으로 판정한다. AS-Norm은 그 뒤 두 경로에 똑같이 걸리므로
게이트 선택에는 영향을 주지 않는다.

실행:
    .venv/bin/python -m eval.separation_gate_eval --max-speakers 20 --per-speaker 3
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.services import embedding as embedding_svc
from app.services import separation as separation_svc
from app.services import vad as vad_svc
from eval import metrics
from eval.dataset import load_utterances
from eval.separation_eval import MIX_SECONDS, _fit, _load, build_trials

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"

#: 훑어볼 게이트 임계값. 원시 코사인 범위에서 실사용 구간을 촘촘히 본다.
GATE_GRID = [round(0.05 * i, 2) for i in range(0, 17)]  # 0.00 ~ 0.80

#: 이보다 작은 EER 악화는 운영상 의미가 없다고 본다.
#:
#: 허용 오차를 해상도로 잡으면 안 된다. 표본이 작을수록 해상도가 커져 **분명히
#: 나쁜 지점까지 통과시키기 때문이다.** 코호트 실험(06 Phase 7)에서 같은 실수를
#: 했다. 해상도는 "구분할 수 있는가"의 기준이지 "허용해도 되는가"의 기준이 아니다.
PRACTICAL_EFFECT = 0.005


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def main(max_speakers: int, per_speaker: int) -> None:
    settings = get_settings()
    rate = settings.target_sample_rate
    logger.info("백엔드=%s 모델=%s", settings.embedding_backend, settings.embedding_model)

    embedding_svc.warmup(
        settings.embedding_model, settings.model_cache_dir,
        backend=settings.embedding_backend, onnx_threads=settings.onnx_intra_op_threads,
    )
    separation_svc.warmup(settings.separation_model, settings.model_cache_dir)

    def embed(samples: np.ndarray) -> np.ndarray | None:
        """서버와 동일하게 VAD → 임베딩. VAD가 반려하면 None."""
        try:
            result = vad_svc.apply(
                samples, rate,
                threshold=settings.vad_threshold,
                min_silence_ms=settings.vad_min_silence_ms,
                speech_pad_ms=settings.vad_speech_pad_ms,
                min_speech_sec=settings.min_speech_sec,
            )
        except Exception:
            return None
        emb = embedding_svc.extract(
            result.samples,
            model_name=settings.embedding_model,
            cache_dir=settings.model_cache_dir,
            backend=settings.embedding_backend,
            onnx_threads=settings.onnx_intra_op_threads,
        )
        return np.asarray(emb.vector, dtype=np.float32)

    def raw_embed(samples: np.ndarray) -> list[float]:
        return embedding_svc.extract(
            samples,
            model_name=settings.embedding_model,
            cache_dir=settings.model_cache_dir,
            backend=settings.embedding_backend,
            onnx_threads=settings.onnx_intra_op_threads,
        ).vector

    utts = load_utterances("dev-clean", max_per_speaker=4, seed=0)
    keep = sorted({u.speaker for u in utts})[:max_speakers]
    utts = [u for u in utts if u.speaker in keep]
    by_key = {u.key: u for u in utts}
    trials = build_trials(utts, rate, per_speaker=per_speaker)
    logger.info("트라이얼 %d건 (화자 %d명)", len(trials), len(keep))

    length = int(rate * MIX_SECONDS)

    # 등록 성문은 트라이얼 간 공유되므로 한 번만 뽑는다.
    enroll_cache: dict[str, np.ndarray] = {}
    for t in trials:
        if t.enroll_key not in enroll_cache:
            vec = embed(_fit(_load(by_key[t.enroll_key].path, rate), length))
            if vec is not None:
                enroll_cache[t.enroll_key] = vec

    enroll_keys = sorted(enroll_cache)
    enroll_m = _normalize_rows(np.stack([enroll_cache[k] for k in enroll_keys]))
    speaker_of = {k: by_key[k].speaker for k in enroll_keys}

    # 검증 임베딩 네 가지를 트라이얼마다 만든다.
    #   깨끗 직접 / 깨끗 분리 / 혼합 직접 / 혼합 분리
    # "깨끗 분리"가 중요하다 — 단일 화자에 분리를 거는 것이 곧 게이트가 막으려는
    # 손해이며, 그 크기를 여기서 잰다.
    rows: list[dict] = []
    started = time.perf_counter()

    for i, t in enumerate(trials, 1):
        enroll = enroll_cache.get(t.enroll_key)
        if enroll is None:
            continue

        variants: dict[str, np.ndarray | None] = {
            "clean_direct": embed(t.clean),
            "mixed_direct": embed(t.mixed),
        }
        for name, signal in (("clean", t.clean), ("mixed", t.mixed)):
            result = separation_svc.extract_target(
                signal, enroll,
                embed_fn=raw_embed,
                model_name=settings.separation_model,
                cache_dir=settings.model_cache_dir,
            )
            variants[f"{name}_separated"] = embed(result.target)

        if any(v is None for v in variants.values()):
            continue

        rows.append({
            "target_speaker": t.target_speaker,
            "enroll_key": t.enroll_key,
            **{k: v for k, v in variants.items()},
        })

        if i % 5 == 0:
            logger.info("트라이얼 %d/%d (%.0f초)", i, len(trials), time.perf_counter() - started)

    if not rows:
        raise SystemExit("유효한 트라이얼이 없다 — VAD가 전부 반려했는지 확인할 것")

    logger.info("검증 임베딩 완성: %d트라이얼 × 4종", len(rows))

    # --- 점수 행렬 ---
    #
    # 각 트라이얼의 검증 임베딩을 모든 등록 성문과 대조한다. 같은 화자면 genuine,
    # 아니면 impostor. 분리 추론은 비싸지만 내적은 싸므로 이렇게 검정력을 키운다.
    labels: list[int] = []
    scores: dict[str, list[float]] = {k: [] for k in
                                      ("clean_direct", "clean_separated",
                                       "mixed_direct", "mixed_separated")}

    for row in rows:
        probes = {k: row[k] / (np.linalg.norm(row[k]) or 1.0) for k in scores}
        sims = {k: enroll_m @ v for k, v in probes.items()}
        for j, key in enumerate(enroll_keys):
            labels.append(1 if speaker_of[key] == row["target_speaker"] else 0)
            for k in scores:
                scores[k].append(float(sims[k][j]))

    y = np.array(labels)
    arr = {k: np.array(v) for k, v in scores.items()}
    resolution = 1 / min((y == 1).sum(), (y == 0).sum())

    # --- 임계값 훑기 ---
    #
    # 게이트는 "직접 점수가 t 이상이면 분리를 건너뛴다"이므로, 게이트가 적용된
    # 점수는 직접 점수와 분리 점수 중 하나를 고른 값이다.
    sweep: list[dict] = []
    for t_gate in GATE_GRID:
        entry: dict = {"gate": t_gate}
        for cond in ("clean", "mixed"):
            direct = arr[f"{cond}_direct"]
            separated = arr[f"{cond}_separated"]
            skip = direct >= t_gate
            gated = np.where(skip, direct, separated)
            m = metrics.compute(gated, y)
            entry[cond] = {
                "eer": m.eer,
                "min_dcf": m.min_dcf,
                # 실제 비용은 genuine 요청 기준으로 본다. 정상 사용자가 분리를
                # 얼마나 자주 겪는지가 체감 지연이다.
                "separation_rate": float((~skip)[y == 1].mean()),
            }
        sweep.append(entry)

    baseline = {}
    for cond in ("clean", "mixed"):
        for mode in ("direct", "separated"):
            m = metrics.compute(arr[f"{cond}_{mode}"], y)
            baseline[f"{cond}_{mode}"] = {"eer": m.eer, "min_dcf": m.min_dcf}

    # --- 결론을 낼 수 있는 측정인지 먼저 확인 ---
    #
    # 게이트 캘리브레이션은 "분리를 건너뛰어도 되는 지점"을 찾는 일이다. 그런데
    # 애초에 분리가 혼합에서 얼마나 도움이 되는지를 이 표본으로 구분하지 못하면,
    # 무엇을 잃는지도 잴 수 없어 어떤 임계값도 근거가 없다.
    warnings: list[str] = []
    mixed_always = baseline["mixed_separated"]["eer"]
    mixed_never = baseline["mixed_direct"]["eer"]
    mixed_gap = mixed_never - mixed_always
    if mixed_gap < resolution:
        warnings.append(
            f"혼합 입력에서 분리의 효과({mixed_gap*100:.2f}%p)가 해상도"
            f"({resolution*100:.2f}%p)보다 작다. 분리가 얼마나 도움이 되는지부터"
            f" 구분할 수 없으므로 게이트 임계값을 정할 근거가 없다."
            f" --max-speakers/--per-speaker를 늘릴 것."
        )

    # --- 권고 임계값 ---
    #
    # 혼합 EER이 "항상 분리"보다 의미 있게 나빠지지 않는 선에서, 깨끗한 입력의
    # 분리 호출을 가장 많이 줄이는 값을 고른다.
    tolerance = min(resolution, PRACTICAL_EFFECT)
    feasible = [
        s for s in sweep
        if s["mixed"]["eer"] <= mixed_always + tolerance
    ]
    recommended = (
        min(feasible, key=lambda s: s["clean"]["separation_rate"])
        if feasible and not warnings else None
    )

    report = {
        "backend": settings.embedding_backend,
        "model": settings.embedding_model,
        "separation_model": settings.separation_model,
        "trials": len(rows),
        "genuine_pairs": int((y == 1).sum()),
        "impostor_pairs": int((y == 0).sum()),
        "eer_resolution": float(resolution),
        "baseline": baseline,
        "sweep": sweep,
        "tolerance": tolerance,
        "recommended_gate": recommended["gate"] if recommended else None,
        "warnings": warnings,
        "conclusive": not warnings,
    }
    path = DATA_DIR / "separation_gate_eval.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n" + "=" * 78)
    print("분리 게이트 임계값 캘리브레이션")
    print(f"genuine {(y==1).sum()}쌍 / impostor {(y==0).sum()}쌍 "
          f"· EER 해상도 {resolution*100:.2f}%p")
    print("-" * 78)
    print("기준선 (게이트 없음)")
    for cond in ("clean", "mixed"):
        d = baseline[f"{cond}_direct"]["eer"] * 100
        s = baseline[f"{cond}_separated"]["eer"] * 100
        label = {"clean": "깨끗한 입력", "mixed": "2인 혼합"}[cond]
        print(f"  {label:<12} 분리 안 함 {d:6.2f}%   항상 분리 {s:6.2f}%")
    print("-" * 78)
    print(f"{'게이트':>7} {'깨끗 EER':>10} {'혼합 EER':>10} "
          f"{'깨끗 분리율':>12} {'혼합 분리율':>12}")
    print("-" * 78)
    for s in sweep:
        mark = " ←권고" if recommended and s["gate"] == recommended["gate"] else ""
        print(f"{s['gate']:>7.2f} {s['clean']['eer']*100:>9.2f}% "
              f"{s['mixed']['eer']*100:>9.2f}% "
              f"{s['clean']['separation_rate']*100:>11.1f}% "
              f"{s['mixed']['separation_rate']*100:>11.1f}%{mark}")
    print("=" * 78)

    if warnings:
        print("\n⚠ 이 측정으로는 임계값을 정할 수 없다:")
        for w in warnings:
            print(f"  - {w}")
    elif recommended:
        g = recommended
        print(f"\n권고 VG_SEPARATION_GATE_SCORE={g['gate']}")
        print(f"  혼합 EER {g['mixed']['eer']*100:.2f}% "
              f"(항상 분리 {mixed_always*100:.2f}%, 허용 오차 {tolerance*100:.2f}%p)")
        print(f"  깨끗한 입력에서 분리를 {(1-g['clean']['separation_rate'])*100:.0f}% 건너뛴다")
        print(f"  깨끗한 입력 EER {baseline['clean_separated']['eer']*100:.2f}% → "
              f"{g['clean']['eer']*100:.2f}%")
    else:
        print("\n혼합 EER을 지키는 임계값이 없다. 게이트 없이 항상 분리할 것.")

    print(f"\n보고서: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="분리 게이트 임계값 캘리브레이션")
    parser.add_argument("--max-speakers", type=int, default=20)
    parser.add_argument("--per-speaker", type=int, default=3)
    args = parser.parse_args()
    main(args.max_speakers, args.per_speaker)
