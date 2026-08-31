"""음성 향상(DeepFilterNet)의 효과 실측.

**향상은 공짜가 아니다.** 이미 깨끗한 오디오에 적용하면 아티팩트를 더해 임베딩
품질을 오히려 떨어뜨릴 수 있다. 그러므로 "향상을 켰더니 좋아졌다"를 가정하지 않고,
깨끗한 조건과 소음 조건 각각에서 EER을 측정해 채택 여부를 데이터로 정한다.

측정 격자:

    조건        | 향상 없음 | 향상 적용
    ------------|-----------|----------
    clean       |     ✓     |    ✓
    SNR 20dB    |     ✓     |    ✓
    SNR 10dB    |     ✓     |    ✓
    SNR  5dB    |     ✓     |    ✓
    SNR  0dB    |     ✓     |    ✓

등록 발화는 항상 깨끗한 것을 쓰고 검증 발화에만 소음을 섞는다. 실제 서비스가
그렇기 때문이다 — 등록은 통제된 환경에서 한 번, 검증은 아무 데서나 한다.

실행:
    .venv/bin/python -m eval.noise_eval                    # 기본 (샘플 축소)
    .venv/bin/python -m eval.noise_eval --max-per-speaker 6
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from app.config import get_settings
from app.services import embedding as embedding_svc
from app.services import enhance as enhance_svc
from app.services import vad as vad_svc
from eval import metrics
from eval.dataset import build_trials, load_utterances

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"

#: 측정할 소음 조건 (None은 소음 없음)
SNR_LEVELS = (None, 20.0, 10.0, 5.0, 0.0)


def _load_audio(path: Path, target_rate: int) -> np.ndarray:
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]
    if rate != target_rate:
        duration = len(mono) / rate
        idx = np.linspace(0, len(mono) - 1, num=int(round(duration * target_rate)))
        mono = np.interp(idx, np.arange(len(mono)), mono).astype(np.float32)
    return np.ascontiguousarray(mono, dtype=np.float32)


def add_noise(clean: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """지정한 SNR로 정상 잡음을 더한다.

    실제 환경 소음(MUSAN 등)이 아니라 가우시안 잡음이다. 광대역 정상 잡음은
    소음 억제가 가장 잘 다루는 종류이므로, 여기서 나온 개선폭은 **낙관적인
    상한**으로 읽어야 한다. 실제 배경 대화·음악에서는 더 어렵다.
    """
    noise = rng.standard_normal(len(clean)).astype(np.float32)
    signal_power = float(np.mean(clean**2))
    noise_power = float(np.mean(noise**2))
    if signal_power <= 0 or noise_power <= 0:
        return clean
    scale = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
    noisy = clean + scale * noise
    peak = float(np.abs(noisy).max())
    # 클리핑을 피한다 — 클리핑 왜곡이 소음 효과와 섞이면 측정이 오염된다.
    # 혼합물 전체에 균일 게인을 걸므로 SNR은 정확히 보존되지만, 반환된 파형과
    # 원본 clean의 차이는 더 이상 순수 잡음이 아니다. SNR을 다시 재려면
    # 스케일 불변 방식(투영 후 잔차)을 써야 한다.
    if peak > 1.0:
        noisy = noisy / peak
    return noisy.astype(np.float32)


def _embed(samples: np.ndarray, settings) -> np.ndarray | None:
    """서버와 동일하게 VAD → 임베딩. VAD가 반려하면 None."""
    try:
        result = vad_svc.apply(
            samples,
            settings.target_sample_rate,
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
    )
    return np.asarray(emb.vector, dtype=np.float32)


def main(max_per_speaker: int, max_speakers: int) -> None:
    settings = get_settings()
    logger.info("백엔드=%s 모델=%s", settings.embedding_backend, settings.embedding_model)

    utts = load_utterances("dev-clean", max_per_speaker=max_per_speaker, seed=0)
    if max_speakers:
        keep = sorted({u.speaker for u in utts})[:max_speakers]
        utts = [u for u in utts if u.speaker in keep]
    logger.info("발화 %d개 / 화자 %d명", len(utts), len({u.speaker for u in utts}))

    embedding_svc.warmup(
        settings.embedding_model, settings.model_cache_dir, backend=settings.embedding_backend
    )
    enhance_svc.warmup()

    # 원본 파형을 한 번만 읽어둔다
    waveforms = {u.key: _load_audio(u.path, settings.target_sample_rate) for u in utts}

    # 등록 임베딩은 깨끗한 원본에서 한 번만 만든다 (실서비스와 같은 조건)
    logger.info("등록 임베딩 추출 (깨끗한 원본)")
    enroll_emb: dict[str, np.ndarray] = {}
    for u in utts:
        e = _embed(waveforms[u.key], settings)
        if e is not None:
            enroll_emb[u.key] = e

    usable = [u for u in utts if u.key in enroll_emb]
    trials = build_trials(usable, genuine_per_speaker=15, impostor_per_speaker=15, seed=1)
    labels = np.array([1 if t.is_same_speaker else 0 for t in trials])
    logger.info("트라이얼 %d건", len(trials))

    results: dict[str, dict] = {}
    rng_seed = 1234

    for snr in SNR_LEVELS:
        cond = "clean" if snr is None else f"snr{int(snr)}"
        for enhanced in (False, True):
            key = f"{cond}/{'enhanced' if enhanced else 'raw'}"
            started = time.perf_counter()

            # 검증 임베딩: 소음(및 향상)을 적용해 다시 추출
            rng = np.random.default_rng(rng_seed)
            test_emb: dict[str, np.ndarray] = {}
            for u in usable:
                wav = waveforms[u.key]
                if snr is not None:
                    wav = add_noise(wav, snr, rng)
                if enhanced:
                    wav = enhance_svc.apply(wav, settings.target_sample_rate)
                e = _embed(wav, settings)
                if e is not None:
                    test_emb[u.key] = e

            # 임베딩 추출에 실패한 발화가 낀 트라이얼은 제외한다
            valid = [
                i
                for i, t in enumerate(trials)
                if t.enroll in enroll_emb and t.test in test_emb
            ]
            if len(valid) < 20:
                logger.warning("%s: 유효 트라이얼 부족 (%d건) — 건너뜀", key, len(valid))
                continue

            a = np.stack([enroll_emb[trials[i].enroll] for i in valid])
            b = np.stack([test_emb[trials[i].test] for i in valid])
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            b = b / np.linalg.norm(b, axis=1, keepdims=True)
            scores = np.sum(a * b, axis=1)
            m = metrics.compute(scores, labels[valid])

            results[key] = {
                "eer": m.eer,
                "eer_threshold": m.eer_threshold,
                "min_dcf": m.min_dcf,
                "separation": m.separation,
                "trials": len(valid),
                "vad_rejected": len(usable) - len(test_emb),
                "elapsed_sec": round(time.perf_counter() - started, 1),
            }
            logger.info(
                "%s: EER=%.2f%%, minDCF=%.4f, 분리도=%.2f, VAD반려=%d, %.0f초",
                key,
                m.eer * 100,
                m.min_dcf,
                m.separation,
                results[key]["vad_rejected"],
                results[key]["elapsed_sec"],
            )

    report_path = DATA_DIR / f"noise_eval__{settings.embedding_backend}.json"
    report_path.write_text(
        json.dumps(
            {
                "backend": settings.embedding_backend,
                "model": settings.embedding_model,
                "speakers": len({u.speaker for u in usable}),
                "trials": len(trials),
                "note": "잡음은 가우시안. 실제 환경 소음보다 억제가 쉬우므로 개선폭은 상한으로 읽을 것.",
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n" + "=" * 76)
    print(f"음성 향상 효과 — {settings.embedding_backend} / {settings.embedding_model}")
    print(f"화자 {len({u.speaker for u in usable})}명, 트라이얼 {len(trials)}건")
    print("-" * 76)
    print(f"{'조건':<12}{'향상없음 EER':>16}{'향상적용 EER':>16}{'변화':>14}{'판정':>12}")
    print("-" * 76)
    for snr in SNR_LEVELS:
        cond = "clean" if snr is None else f"SNR {int(snr)}dB"
        raw_key = f"{'clean' if snr is None else f'snr{int(snr)}'}/raw"
        enh_key = f"{'clean' if snr is None else f'snr{int(snr)}'}/enhanced"
        if raw_key not in results or enh_key not in results:
            continue
        r, e = results[raw_key]["eer"] * 100, results[enh_key]["eer"] * 100
        delta = e - r
        verdict = "개선" if delta < -0.01 else ("악화" if delta > 0.01 else "동일")
        print(f"{cond:<14}{r:>13.2f}%{e:>15.2f}%{delta:>+13.2f}%p{verdict:>12}")
    print("=" * 76)
    print(f"\n보고서: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="음성 향상 효과 실측")
    parser.add_argument("--max-per-speaker", type=int, default=4)
    parser.add_argument("--max-speakers", type=int, default=20)
    args = parser.parse_args()
    main(args.max_per_speaker, args.max_speakers)
