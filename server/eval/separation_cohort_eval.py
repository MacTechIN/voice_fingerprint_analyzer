"""분리 아티팩트에 맞춘 AS-Norm 코호트 실험 (Phase 7 후속).

**목표: 분리 경로의 잔여 격차 3.12%p를 줄인다.**

Phase 7 실측에서 2인 혼합을 분리해 검증하면 EER이 6.25%였고, 깨끗한 조건
(3.12%)과의 격차가 분리 아티팩트로 잃는 양이었다. 02 §4.3이 예고한 임베딩
왜곡이다.

원래 계획은 **아티팩트 증강 데이터로 임베딩 백본을 파인튜닝**하는 것이었으나,
251화자 ArcFace 학습을 CPU로 하면 수일이 걸려 이 환경에서 비현실적이다.

대신 **학습 없이 같은 문제를 겨냥하는 방법**을 시험한다: AS-Norm 코호트를
분리를 거친 오디오로 만든다. 정규화는 "이 점수가 임포스터 분포에서 얼마나
튀는가"를 재는데, 코호트가 깨끗한 오디오면 **검증 임베딩과 다른 조건**의
분포를 기준으로 삼게 된다. 조건을 맞추면 정규화가 아티팩트로 인한 계통적
치우침을 흡수할 수 있다.

측정 격자 — 검증 조건 × 코호트 종류:

    검증 조건      | 정규화 없음 | 깨끗한 코호트 | 분리 코호트
    ---------------|-------------|---------------|-------------
    clean          |      ✓      |       ✓       |     ✓
    separated      |      ✓      |       ✓       |     ✓

**분리 코호트가 도움이 되지 않을 수도 있다.** DeepFilterNet도 "당연히 좋아질
것"이라 여겼다가 실측에서 악화됐다. 가정하지 않고 잰다.

실행:
    .venv/bin/python -m eval.separation_cohort_eval --max-speakers 20
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

#: 코호트 상위 K개. Phase 6에서 정한 운영값과 같게 둔다.
#:
#: **코호트가 이보다 작으면 AS-Norm이 아니게 된다.** 상위 K를 적응적으로 고르는
#: 것이 AS-Norm의 핵심인데, K가 코호트 크기 이상이면 전체를 쓰게 되어 그냥
#: 평균·표준편차 정규화가 된다. 아래에서 이 조건을 검사한다.
TOP_K = 200

#: 코호트가 top_k의 몇 배는 되어야 적응적 선택이 의미를 갖는다.
MIN_COHORT_RATIO = 1.5


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _asnorm(
    raw: np.ndarray,
    enroll_vs_cohort: np.ndarray,
    test_vs_cohort: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """AS-Norm 대칭 정규화 (배치)."""
    k = min(top_k, enroll_vs_cohort.shape[1])
    e_top = np.sort(enroll_vs_cohort, axis=1)[:, -k:]
    t_top = np.sort(test_vs_cohort, axis=1)[:, -k:]
    e_mean, e_std = e_top.mean(axis=1), np.maximum(e_top.std(axis=1), 1e-6)
    t_mean, t_std = t_top.mean(axis=1), np.maximum(t_top.std(axis=1), 1e-6)
    return 0.5 * ((raw - e_mean) / e_std + (raw - t_mean) / t_std)


def main(max_speakers: int, per_speaker: int, cohort_speakers: int) -> None:
    settings = get_settings()
    rate = settings.target_sample_rate
    logger.info("백엔드=%s 모델=%s", settings.embedding_backend, settings.embedding_model)

    embedding_svc.warmup(
        settings.embedding_model, settings.model_cache_dir,
        backend=settings.embedding_backend, onnx_threads=settings.onnx_intra_op_threads,
    )
    separation_svc.warmup(settings.separation_model, settings.model_cache_dir)

    def embed(samples: np.ndarray, *, use_vad: bool = True) -> np.ndarray | None:
        if use_vad:
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
            samples = result.samples
        emb = embedding_svc.extract(
            samples,
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

    # --- 평가 트라이얼 (dev-clean) ---
    utts = load_utterances("dev-clean", max_per_speaker=4, seed=0)
    keep = sorted({u.speaker for u in utts})[:max_speakers]
    utts = [u for u in utts if u.speaker in keep]
    by_key = {u.key: u for u in utts}
    trials = build_trials(utts, rate, per_speaker=per_speaker)
    logger.info("트라이얼 %d건 (화자 %d명)", len(trials), len(keep))

    # --- 코호트 (test-clean, 평가 화자와 분리) ---
    cohort_utts = load_utterances("test-clean", max_per_speaker=3, seed=0)
    cohort_keep = sorted({u.speaker for u in cohort_utts})[:cohort_speakers]
    cohort_utts = [u for u in cohort_utts if u.speaker in cohort_keep]
    logger.info("코호트 후보 %d개 (화자 %d명)", len(cohort_utts), len(cohort_keep))

    length = int(rate * MIX_SECONDS)
    rng = np.random.default_rng(11)

    clean_cohort: list[np.ndarray] = []
    separated_cohort: list[np.ndarray] = []
    started = time.perf_counter()

    for i, utt in enumerate(cohort_utts, 1):
        clean = _fit(_load(utt.path, rate), length)
        clean = clean / (np.abs(clean).max() or 1) * 0.7

        vec = embed(clean)
        if vec is not None:
            clean_cohort.append(vec)

        # 분리 코호트: 같은 발화를 다른 화자와 섞어 분리한 뒤 임베딩.
        # 검증 경로와 **같은 처리를 거친** 임베딩이어야 조건이 맞는다.
        other = rng.choice([u for u in cohort_utts if u.speaker != utt.speaker])
        interferer = _fit(_load(other.path, rate), length)
        interferer = interferer / (np.abs(interferer).max() or 1) * 0.7
        mixed = clean + interferer
        peak = np.abs(mixed).max()
        if peak > 1.0:
            mixed = mixed / peak * 0.95

        result = separation_svc.extract_target(
            mixed.astype(np.float32), vec if vec is not None else clean[:1].tolist(),
            embed_fn=raw_embed,
            model_name=settings.separation_model,
            cache_dir=settings.model_cache_dir,
        )
        sep_vec = embed(result.target)
        if sep_vec is not None:
            separated_cohort.append(sep_vec)

        if i % 10 == 0:
            logger.info("코호트 %d/%d (%.0f초)", i, len(cohort_utts), time.perf_counter() - started)

    logger.info("코호트 완성: 깨끗 %d개, 분리 %d개", len(clean_cohort), len(separated_cohort))

    # --- 트라이얼 임베딩 ---
    enroll_vecs: list[np.ndarray] = []
    clean_probes: list[np.ndarray] = []
    separated_probes: list[np.ndarray] = []
    labels: list[int] = []

    enroll_cache: dict[str, np.ndarray] = {}
    for t in trials:
        if t.enroll_key not in enroll_cache:
            vec = embed(_fit(_load(by_key[t.enroll_key].path, rate), length))
            if vec is not None:
                enroll_cache[t.enroll_key] = vec

    for i, t in enumerate(trials, 1):
        enroll = enroll_cache.get(t.enroll_key)
        if enroll is None:
            continue

        clean_probe = embed(t.clean)
        sep_result = separation_svc.extract_target(
            t.mixed, enroll, embed_fn=raw_embed,
            model_name=settings.separation_model, cache_dir=settings.model_cache_dir,
        )
        sep_probe = embed(sep_result.target)
        if clean_probe is None or sep_probe is None:
            continue

        # genuine 1건 + **다른 모든 화자를 impostor로** 사용한다.
        #
        # 분리 추론이 트라이얼당 약 7초로 비싼 반면, 같은 probe를 여러 등록
        # 임베딩과 대조하는 것은 내적 한 번이다. impostor를 1건만 쓰면 EER
        # 해상도가 1/트라이얼수로 묶여(40건이면 2.5%p) 코호트 차이를 잴 수 없다.
        enroll_vecs.append(enroll)
        clean_probes.append(clean_probe)
        separated_probes.append(sep_probe)
        labels.append(1)

        for key, vec in enroll_cache.items():
            if by_key[key].speaker == t.target_speaker:
                continue
            enroll_vecs.append(vec)
            clean_probes.append(clean_probe)
            separated_probes.append(sep_probe)
            labels.append(0)

        if i % 10 == 0:
            logger.info("트라이얼 %d/%d", i, len(trials))

    y = np.array(labels)
    enroll_m = _normalize_rows(np.stack(enroll_vecs))
    probes = {
        "clean": _normalize_rows(np.stack(clean_probes)),
        "separated": _normalize_rows(np.stack(separated_probes)),
    }
    cohorts = {
        "none": None,
        "clean": _normalize_rows(np.stack(clean_cohort)) if clean_cohort else None,
        "separated": _normalize_rows(np.stack(separated_cohort)) if separated_cohort else None,
    }

    # --- 측정이 결론을 낼 수 있는 조건인지 먼저 확인 ---
    #
    # 이 실험은 조건 간 EER 차이를 보는 것이 목적이다. 표본이 적어 해상도가
    # 차이보다 크면 어떤 숫자가 나오든 해석할 수 없다. 그런 상태로 결과를
    # 보고하면 없는 효과를 있다고 읽게 된다.
    warnings: list[str] = []
    smallest_cohort = min(
        len(c) for c in (clean_cohort, separated_cohort) if c
    ) if (clean_cohort or separated_cohort) else 0
    if smallest_cohort < TOP_K * MIN_COHORT_RATIO:
        warnings.append(
            f"코호트({smallest_cohort}개)가 top_k({TOP_K})에 비해 작다. "
            f"상위 K 적응 선택이 사실상 무력화되어 운영 설정(코호트 310/K 200)과 "
            f"다른 것을 재게 된다. --cohort-speakers를 늘리거나 TOP_K를 낮출 것."
        )

    results: dict[str, dict] = {}
    for probe_name, probe_m in probes.items():
        raw = np.sum(enroll_m * probe_m, axis=1)
        results[probe_name] = {}
        for cohort_name, cohort_m in cohorts.items():
            if cohort_name == "none":
                scores = raw
            elif cohort_m is None:
                continue
            else:
                scores = _asnorm(raw, enroll_m @ cohort_m.T, probe_m @ cohort_m.T, TOP_K)
            m = metrics.compute(scores, y)
            results[probe_name][cohort_name] = {
                "eer": m.eer,
                "min_dcf": m.min_dcf,
                "separation": m.separation,
            }

    # 관측된 차이가 해상도보다 작으면 결론을 낼 수 없다
    resolution = 1 / min((y == 1).sum(), (y == 0).sum())
    sep_row = results.get("separated", {})
    if len(sep_row) >= 2:
        eers = [v["eer"] for v in sep_row.values()]
        spread = max(eers) - min(eers)
        if spread < resolution:
            warnings.append(
                f"분리 조건의 EER 차이({spread*100:.2f}%p)가 해상도"
                f"({resolution*100:.2f}%p)보다 작다. **어느 쪽이 낫다고 말할 수 없다.** "
                f"--per-speaker를 늘려 genuine 트라이얼을 확보할 것."
            )

    report = {
        "backend": settings.embedding_backend,
        "model": settings.embedding_model,
        "separation_model": settings.separation_model,
        "genuine_trials": int((y == 1).sum()),
        "impostor_trials": int((y == 0).sum()),
        "eer_resolution": float(1 / min((y == 1).sum(), (y == 0).sum())),
        "cohort_size": {"clean": len(clean_cohort), "separated": len(separated_cohort)},
        "top_k": TOP_K,
        "results": results,
        "warnings": warnings,
        "conclusive": not warnings,
    }
    path = DATA_DIR / "separation_cohort_eval.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n" + "=" * 72)
    print("분리 경로 AS-Norm 코호트 비교")
    resolution = 1 / min((y == 1).sum(), (y == 0).sum())
    print(f"genuine {(y==1).sum()}건 / impostor {(y==0).sum()}건 "
          f"· 코호트 깨끗 {len(clean_cohort)}개 / 분리 {len(separated_cohort)}개")
    print(f"EER 해상도 {resolution*100:.2f}%p — 이보다 작은 차이는 구분할 수 없다")
    print("-" * 72)
    print(f"{'검증 조건':<14}{'정규화 없음':>14}{'깨끗한 코호트':>16}{'분리 코호트':>14}")
    print("-" * 72)
    for probe_name, row in results.items():
        label = {"clean": "깨끗", "separated": "분리+선택"}[probe_name]
        cells = "".join(
            f"{row[c]['eer']*100:>13.2f}%" if c in row else f"{'—':>14}"
            for c in ("none", "clean", "separated")
        )
        print(f"{label:<15}{cells}")
    print("-" * 72)
    sep = results.get("separated", {})
    if "clean" in sep and "separated" in sep:
        a, b = sep["clean"]["eer"], sep["separated"]["eer"]
        delta = (a - b) / a * 100 if a else 0
        verdict = "개선" if b < a else ("악화" if b > a else "동일")
        print(f"분리 경로에서 코호트를 맞춘 효과: EER {a*100:.2f}% → {b*100:.2f}% ({delta:+.1f}%, {verdict})")
    print("=" * 72)
    if warnings:
        print("\n⚠ 이 측정은 결론을 낼 수 없다:")
        for w in warnings:
            print(f"  - {w}")
    print(f"\n보고서: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="분리 아티팩트 코호트 실험")
    parser.add_argument("--max-speakers", type=int, default=20)
    parser.add_argument("--per-speaker", type=int, default=2)
    parser.add_argument("--cohort-speakers", type=int, default=30)
    args = parser.parse_args()
    main(args.max_speakers, args.per_speaker, args.cohort_speakers)
