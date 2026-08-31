"""임계값 캘리브레이션 실행 스크립트.

원시 코사인과 AS-Norm 정규화 점수 각각에 대해 EER·minDCF를 측정하고,
운영 임계값 후보를 산출한다.

실행:
    .venv/bin/python -m eval.calibrate

평가 화자(dev-clean)와 코호트 화자(test-clean)는 분리되어 있다. 섞이면 정규화가
자기 자신을 참조하게 되어 측정이 무의미해진다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.services.asnorm import CohortIndex
from eval import metrics
from eval.dataset import build_trials, load_utterances
from eval.extract import extract_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / ".data" / "cache"
REPORT_PATH = Path(__file__).resolve().parent.parent / ".data" / "calibration_report.json"

# 코호트 상위 K개. 값이 클수록 통계가 안정되지만 '가장 혼동하기 쉬운 임포스터'라는
# 적응적 성격이 옅어진다. 문헌에서 통상 200~400을 쓰며, 여기서는 코호트 크기에
# 맞춰 몇 가지를 비교한다.
TOP_K_CANDIDATES = (50, 100, 200, 300)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def main() -> None:
    # --- 1. 평가 트라이얼 ---
    logger.info("평가 발화 로드 (dev-clean)")
    eval_utts = load_utterances("dev-clean", max_per_speaker=8, seed=0)
    logger.info("평가 발화 %d개 / 화자 %d명", len(eval_utts), len({u.speaker for u in eval_utts}))

    eval_emb = extract_all(eval_utts, CACHE_DIR / "dev-clean.npz")

    # 임베딩 추출에 실패(VAD 반려)한 발화는 트라이얼에서 제외한다
    usable = [u for u in eval_utts if u.key in eval_emb]
    trials = build_trials(usable, genuine_per_speaker=30, impostor_per_speaker=30, seed=1)
    logger.info("트라이얼 %d건 생성", len(trials))

    # --- 2. 코호트 ---
    logger.info("코호트 발화 로드 (test-clean, 평가 화자와 분리)")
    cohort_utts = load_utterances("test-clean", max_per_speaker=8, seed=0)
    cohort_emb = extract_all(cohort_utts, CACHE_DIR / "test-clean.npz")
    cohort_matrix = np.stack([cohort_emb[u.key] for u in cohort_utts if u.key in cohort_emb])
    logger.info("코호트 %d개 임베딩", len(cohort_matrix))

    # --- 3. 원시 코사인 점수 ---
    enroll_vecs = _normalize_rows(np.stack([eval_emb[t.enroll] for t in trials]))
    test_vecs = _normalize_rows(np.stack([eval_emb[t.test] for t in trials]))
    raw_scores = np.sum(enroll_vecs * test_vecs, axis=1)
    labels = np.array([1 if t.is_same_speaker else 0 for t in trials])

    raw_metrics = metrics.compute(raw_scores, labels)
    logger.info(
        "원시 코사인: EER=%.2f%% (thr=%.4f), minDCF=%.4f, 분리도=%.2f",
        raw_metrics.eer * 100,
        raw_metrics.eer_threshold,
        raw_metrics.min_dcf,
        raw_metrics.separation,
    )

    # --- 4. AS-Norm ---
    cohort_norm = _normalize_rows(cohort_matrix)
    # 트라이얼별로 코호트 전체와의 유사도를 한 번에 구한다
    enroll_vs_cohort = enroll_vecs @ cohort_norm.T
    test_vs_cohort = test_vecs @ cohort_norm.T

    asnorm_results = {}
    for top_k in TOP_K_CANDIDATES:
        k = min(top_k, cohort_norm.shape[0])
        e_top = np.sort(enroll_vs_cohort, axis=1)[:, -k:]
        t_top = np.sort(test_vs_cohort, axis=1)[:, -k:]

        e_mean, e_std = e_top.mean(axis=1), np.maximum(e_top.std(axis=1), 1e-6)
        t_mean, t_std = t_top.mean(axis=1), np.maximum(t_top.std(axis=1), 1e-6)

        normed = 0.5 * ((raw_scores - e_mean) / e_std + (raw_scores - t_mean) / t_std)
        m = metrics.compute(normed, labels)
        asnorm_results[top_k] = m
        logger.info(
            "AS-Norm (K=%d): EER=%.2f%% (thr=%.4f), minDCF=%.4f, 분리도=%.2f",
            k,
            m.eer * 100,
            m.eer_threshold,
            m.min_dcf,
            m.separation,
        )

    best_k = min(asnorm_results, key=lambda k: asnorm_results[k].eer)
    best = asnorm_results[best_k]

    # --- 5. 현재 운영 임계값 점검 ---
    far_now, frr_now = metrics.error_rates_at(raw_scores, labels, 0.25)
    logger.info(
        "현재 기본 임계값 0.25 (미캘리브레이션): FAR=%.2f%%, FRR=%.2f%%",
        far_now * 100,
        frr_now * 100,
    )

    # --- 6. 보고서 ---
    def dump(m: metrics.Metrics) -> dict:
        return {
            "eer": m.eer,
            "eer_threshold": m.eer_threshold,
            "min_dcf": m.min_dcf,
            "min_dcf_threshold": m.min_dcf_threshold,
            "genuine_mean": m.genuine_mean,
            "genuine_std": m.genuine_std,
            "impostor_mean": m.impostor_mean,
            "impostor_std": m.impostor_std,
            "separation": m.separation,
            "n_genuine": m.n_genuine,
            "n_impostor": m.n_impostor,
        }

    report = {
        "dataset": {
            "eval_split": "dev-clean",
            "cohort_split": "test-clean",
            "eval_speakers": len({u.speaker for u in usable}),
            "eval_utterances": len(usable),
            "cohort_embeddings": int(cohort_norm.shape[0]),
            "trials": len(trials),
        },
        "raw_cosine": dump(raw_metrics),
        "as_norm": {str(k): dump(v) for k, v in asnorm_results.items()},
        "best_top_k": best_k,
        "uncalibrated_threshold_0_25": {"far": far_now, "frr": frr_now},
        "recommended": {
            "raw_threshold": raw_metrics.eer_threshold,
            "asnorm_top_k": best_k,
            "asnorm_threshold": best.eer_threshold,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n" + "=" * 72)
    print(f"평가: {report['dataset']['eval_speakers']}화자 / "
          f"{report['dataset']['trials']}트라이얼 "
          f"(genuine {raw_metrics.n_genuine}, impostor {raw_metrics.n_impostor})")
    print(f"코호트: {report['dataset']['cohort_embeddings']}개 (평가 화자와 분리)")
    print("-" * 72)
    print(f"{'방식':<22}{'EER':>10}{'임계값':>12}{'minDCF':>10}{'분리도':>10}")
    print("-" * 72)
    print(f"{'원시 코사인':<20}{raw_metrics.eer * 100:>9.2f}%"
          f"{raw_metrics.eer_threshold:>12.4f}{raw_metrics.min_dcf:>10.4f}"
          f"{raw_metrics.separation:>10.2f}")
    for k, m in asnorm_results.items():
        label = f"AS-Norm (K={k})"
        print(f"{label:<22}{m.eer * 100:>9.2f}%{m.eer_threshold:>12.4f}"
              f"{m.min_dcf:>10.4f}{m.separation:>10.2f}")
    print("-" * 72)
    improvement = (raw_metrics.eer - best.eer) / raw_metrics.eer * 100 if raw_metrics.eer else 0.0
    print(f"AS-Norm 최적 K={best_k}: EER {raw_metrics.eer * 100:.2f}% → "
          f"{best.eer * 100:.2f}% ({improvement:+.1f}%)")
    print(f"미캘리브레이션 임계값 0.25 → FAR {far_now * 100:.2f}%, FRR {frr_now * 100:.2f}%")
    print(f"권장 원시 임계값: {raw_metrics.eer_threshold:.4f}")
    print(f"권장 AS-Norm 임계값: {best.eer_threshold:.4f} (K={best_k})")
    print("=" * 72)
    print(f"\n보고서: {REPORT_PATH}")


if __name__ == "__main__":
    main()
