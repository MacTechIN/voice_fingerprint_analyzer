"""음성 분리가 화자 검증을 실제로 회복시키는지 측정 (06 Phase 7 Vertical Slice).

**분리가 도움이 된다고 가정하지 않는다.** 02 §4.3은 분리망이 만드는 아티팩트가
임베딩을 왜곡해 검증 성능을 떨어뜨릴 수 있다고 경고했다. 그러므로 세 조건을
같은 트라이얼로 재서 비교한다.

    조건            | 검증 입력
    ----------------|--------------------------------------------------
    clean           | 타겟 화자의 깨끗한 발화 (상한선)
    mixed           | 2인 혼합 그대로 (분리 없이 — 하한선)
    separated       | 혼합 → 분리 → 등록 성문과 가장 맞는 출력 선택

`separated`가 `mixed`보다 나아야 분리가 값을 한 것이고, `clean`에 얼마나
근접하는지가 분리 아티팩트로 잃는 양이다.

등록은 항상 깨끗한 발화로 한다. 실서비스가 그렇다 — 등록은 통제된 환경에서
한 번, 검증은 아무 데서나.

실행:
    .venv/bin/python -m eval.separation_eval --max-speakers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from app.config import get_settings
from app.services import embedding as embedding_svc
from app.services import separation as separation_svc
from app.services import vad as vad_svc
from eval import metrics
from eval.dataset import Utterance, load_utterances

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"

#: 혼합 길이. 너무 짧으면 분리망이 문맥을 못 잡고, 너무 길면 평가가 느려진다.
MIX_SECONDS = 6.0


@dataclass
class Trial:
    """혼합 검증 시도 하나."""

    target_speaker: str
    interferer_speaker: str
    enroll_key: str
    target_key: str
    clean: np.ndarray
    mixed: np.ndarray


def _load(path: Path, rate: int) -> np.ndarray:
    samples, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]
    if sr != rate:
        idx = np.linspace(0, len(mono) - 1, num=int(round(len(mono) / sr * rate)))
        mono = np.interp(idx, np.arange(len(mono)), mono).astype(np.float32)
    return np.ascontiguousarray(mono, dtype=np.float32)


def _fit(samples: np.ndarray, length: int) -> np.ndarray:
    """길이를 맞춘다 — 짧으면 반복해 채우고, 길면 자른다."""
    if len(samples) >= length:
        return samples[:length]
    repeats = int(np.ceil(length / max(len(samples), 1)))
    return np.tile(samples, repeats)[:length]


def build_trials(
    utterances: list[Utterance], rate: int, *, per_speaker: int = 2, seed: int = 7
) -> list[Trial]:
    """화자별로 (등록용, 타겟용) 발화를 뽑고 다른 화자를 간섭으로 섞는다."""
    rng = np.random.default_rng(seed)
    by_speaker: dict[str, list[Utterance]] = {}
    for u in utterances:
        by_speaker.setdefault(u.speaker, []).append(u)

    speakers = sorted(s for s, items in by_speaker.items() if len(items) >= 2)
    length = int(rate * MIX_SECONDS)
    trials: list[Trial] = []

    for target in speakers:
        others = [s for s in speakers if s != target]
        if not others:
            continue
        for _ in range(per_speaker):
            enroll_utt, target_utt = rng.choice(
                np.array(by_speaker[target], dtype=object), size=2, replace=False
            )
            interferer = str(rng.choice(others))
            interferer_utt = rng.choice(np.array(by_speaker[interferer], dtype=object))

            clean = _fit(_load(target_utt.path, rate), length)
            other = _fit(_load(interferer_utt.path, rate), length)

            # 두 화자를 같은 세기로 섞는다 (0dB SIR). 가장 어려운 조건이며,
            # 한쪽이 크면 분리 없이도 풀리는 쉬운 문제가 된다.
            clean_n = clean / (np.abs(clean).max() or 1) * 0.7
            other_n = other / (np.abs(other).max() or 1) * 0.7
            mixed = clean_n + other_n
            peak = np.abs(mixed).max()
            if peak > 1.0:
                mixed = mixed / peak * 0.95

            trials.append(
                Trial(
                    target_speaker=target,
                    interferer_speaker=interferer,
                    enroll_key=enroll_utt.key,
                    target_key=target_utt.key,
                    clean=clean_n.astype(np.float32),
                    mixed=mixed.astype(np.float32),
                )
            )
    return trials


def main(max_speakers: int, per_speaker: int) -> None:
    settings = get_settings()
    rate = settings.target_sample_rate
    logger.info("백엔드=%s 모델=%s", settings.embedding_backend, settings.embedding_model)

    embedding_svc.warmup(
        settings.embedding_model, settings.model_cache_dir,
        backend=settings.embedding_backend, onnx_threads=settings.onnx_intra_op_threads,
    )
    separation_svc.warmup(cache_dir=settings.model_cache_dir)

    utts = load_utterances("dev-clean", max_per_speaker=4, seed=0)
    keep = sorted({u.speaker for u in utts})[:max_speakers]
    utts = [u for u in utts if u.speaker in keep]
    by_key = {u.key: u for u in utts}

    trials = build_trials(utts, rate, per_speaker=per_speaker)
    logger.info("트라이얼 %d건 (화자 %d명)", len(trials), len(keep))

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
        """분리 출력 선택용 — VAD 없이 임베딩만 뽑는다.

        분리 직후 파형에 VAD를 걸면 반려될 수 있는데, 그러면 타겟 선택 자체가
        불가능해진다. 선택은 상대 비교이므로 VAD 없이도 성립한다.
        """
        emb = embedding_svc.extract(
            samples,
            model_name=settings.embedding_model,
            cache_dir=settings.model_cache_dir,
            backend=settings.embedding_backend,
            onnx_threads=settings.onnx_intra_op_threads,
        )
        return emb.vector

    # 등록 임베딩 (깨끗한 발화)
    logger.info("등록 임베딩 추출")
    enroll_cache: dict[str, np.ndarray] = {}
    for t in trials:
        if t.enroll_key in enroll_cache:
            continue
        vec = embed(_fit(_load(by_key[t.enroll_key].path, rate), int(rate * MIX_SECONDS)))
        if vec is not None:
            enroll_cache[t.enroll_key] = vec

    conditions = ("clean", "mixed", "separated")
    scores: dict[str, list[float]] = {c: [] for c in conditions}
    labels: dict[str, list[int]] = {c: [] for c in conditions}
    margins: list[float] = []
    correct_selection = 0
    selection_total = 0
    timings: dict[str, float] = {c: 0.0 for c in conditions}
    skipped = 0

    for i, t in enumerate(trials, 1):
        enroll = enroll_cache.get(t.enroll_key)
        if enroll is None:
            skipped += 1
            continue
        enroll_unit = enroll / (np.linalg.norm(enroll) or 1)

        # 같은 트라이얼에서 다른 화자의 등록 임베딩을 impostor로 쓴다 —
        # genuine/impostor를 같은 오디오 조건에서 비교해야 공정하다
        impostor_key = next(
            (k for k, v in enroll_cache.items()
             if by_key[k].speaker != t.target_speaker), None
        )
        impostor = enroll_cache.get(impostor_key) if impostor_key else None
        if impostor is None:
            skipped += 1
            continue
        impostor_unit = impostor / (np.linalg.norm(impostor) or 1)

        for condition in conditions:
            started = time.perf_counter()
            if condition == "clean":
                probe = embed(t.clean)
            elif condition == "mixed":
                probe = embed(t.mixed)
            else:
                result = separation_svc.extract_target(
                    t.mixed, enroll, embed_fn=raw_embed,
                    cache_dir=settings.model_cache_dir,
                )
                margin = result.selection_margin
                if margin is not None:
                    margins.append(margin)
                probe = embed(result.target)
            timings[condition] += time.perf_counter() - started

            if probe is None:
                continue
            probe_unit = probe / (np.linalg.norm(probe) or 1)
            scores[condition].append(float(enroll_unit @ probe_unit))
            labels[condition].append(1)
            scores[condition].append(float(impostor_unit @ probe_unit))
            labels[condition].append(0)

        # 분리 선택이 옳았는지: 타겟 유사도가 간섭 화자보다 높아야 한다
        selection_total += 1
        if scores["separated"] and len(scores["separated"]) >= 2:
            if scores["separated"][-2] > scores["separated"][-1]:
                correct_selection += 1

        if i % 10 == 0:
            logger.info("진행 %d/%d", i, len(trials))

    results = {}
    for condition in conditions:
        s = np.array(scores[condition])
        y = np.array(labels[condition])
        if len(s) < 4 or y.sum() == 0 or (1 - y).sum() == 0:
            continue
        m = metrics.compute(s, y)
        results[condition] = {
            "eer": m.eer,
            "eer_threshold": m.eer_threshold,
            "min_dcf": m.min_dcf,
            "separation": m.separation,
            "genuine_mean": m.genuine_mean,
            "impostor_mean": m.impostor_mean,
            "trials": len(s) // 2,
            "elapsed_sec": round(timings[condition], 1),
        }

    report = {
        "backend": settings.embedding_backend,
        "model": settings.embedding_model,
        "separation_model": separation_svc.DEFAULT_MODEL,
        "speakers": len(keep),
        "trials": len(trials),
        "skipped": skipped,
        "selection_accuracy": (
            correct_selection / selection_total if selection_total else None
        ),
        "selection_margin_mean": float(np.mean(margins)) if margins else None,
        "conditions": results,
        "note": "0dB SIR 동일 세기 2인 혼합. 실제 환경보다 어려운 조건이다.",
    }
    path = DATA_DIR / f"separation_eval__{settings.embedding_backend}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n" + "=" * 72)
    print(f"음성 분리 효과 — {settings.embedding_model.split('/')[-1]}")
    print(f"분리 모델: {separation_svc.DEFAULT_MODEL.split('/')[-1]}")
    print(f"화자 {len(keep)}명, 혼합 트라이얼 {len(trials)}건 (0dB SIR)")
    print("-" * 72)
    print(f"{'조건':<14}{'EER':>10}{'minDCF':>10}{'분리도':>10}{'genuine':>10}{'소요':>9}")
    print("-" * 72)
    for condition in conditions:
        r = results.get(condition)
        if not r:
            continue
        label = {"clean": "깨끗 (상한)", "mixed": "혼합 (하한)", "separated": "분리+선택"}[condition]
        print(f"{label:<15}{r['eer']*100:>9.2f}%{r['min_dcf']:>10.4f}"
              f"{r['separation']:>10.2f}{r['genuine_mean']:>10.3f}{r['elapsed_sec']:>8.0f}s")
    print("-" * 72)
    if "mixed" in results and "separated" in results:
        m, s = results["mixed"]["eer"], results["separated"]["eer"]
        c = results.get("clean", {}).get("eer")
        delta = (m - s) / m * 100 if m else 0
        print(f"분리 효과: EER {m*100:.2f}% → {s*100:.2f}% ({delta:+.1f}%)")
        if c is not None:
            gap = (s - c) * 100
            print(f"깨끗 대비 잔여 격차: {gap:+.2f}%p (분리 아티팩트로 잃는 양)")
    if report["selection_accuracy"] is not None:
        print(f"타겟 선택 정확도: {report['selection_accuracy']:.1%} "
              f"(평균 마진 {report['selection_margin_mean']:.3f})")
    print("=" * 72)
    print(f"\n보고서: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="음성 분리 효과 실측")
    parser.add_argument("--max-speakers", type=int, default=16)
    parser.add_argument("--per-speaker", type=int, default=2)
    args = parser.parse_args()
    main(args.max_speakers, args.per_speaker)
