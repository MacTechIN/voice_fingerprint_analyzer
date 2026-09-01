"""서버 성능 벤치마크 (Phase 5).

**추측하지 않고 잰다.** Phase 7에서 "스레드 문제일 것"이라 추측했다가 두 번
헛짚었고, 실제 원인은 silero-vad의 전역 스레드 변경이었다. 최적화는 측정에서
시작한다.

측정 항목:

1. **단계별 소요** — 디코딩·VAD·임베딩·스코어링 중 어디가 오래 걸리는가
2. **동시성** — 요청이 겹칠 때 지연이 어떻게 늘어나는가 (CPU 바운드 추론의
   스레드 과다 할당은 처리량을 오히려 떨어뜨린다)
3. **페이로드** — 업로드·응답 크기가 네트워크 지연에 얼마나 기여하는가

실행:
    .venv/bin/python -m eval.bench --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"


@dataclass
class Timing:
    label: str
    samples: list[float]

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0

    @property
    def p50(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0

    @property
    def p95(self) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        return ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]


def _load_speech(seconds: float, rate: int = 16_000) -> np.ndarray:
    """LibriSpeech 발화를 원하는 길이로 준비한다."""
    root = DATA_DIR / "LibriSpeech" / "dev-clean"
    if not root.exists():
        raise SystemExit(f"{root} 가 없습니다. eval/README.md 참조.")
    path = sorted(root.rglob("*.flac"))[0]
    samples, sr = sf.read(path, dtype="float32")
    if sr != rate:
        idx = np.linspace(0, len(samples) - 1, num=int(len(samples) / sr * rate))
        samples = np.interp(idx, np.arange(len(samples)), samples).astype(np.float32)
    need = int(rate * seconds)
    if len(samples) < need:
        samples = np.tile(samples, int(np.ceil(need / len(samples))))
    return np.ascontiguousarray(samples[:need], dtype=np.float32)


def encode(samples: np.ndarray, fmt: str, rate: int = 16_000) -> bytes:
    """지정 포맷으로 인코딩한다.

    FLAC은 무손실이라 임베딩 품질에 영향이 없으면서 WAV보다 작다. 네트워크가
    느린 모바일 환경에서 업로드 시간을 줄일 수 있는지 보려는 것이다.
    """
    buf = io.BytesIO()
    subtype = "PCM_16" if fmt.upper() == "WAV" else None
    sf.write(buf, samples, rate, format=fmt.upper(), subtype=subtype)
    return buf.getvalue()


def stage_breakdown(seconds: float, repeats: int) -> dict:
    """파이프라인 단계별 소요 시간 (서버 내부, 네트워크 제외)."""
    from app.config import get_settings
    from app.services import audio as audio_svc
    from app.services import embedding as embedding_svc
    from app.services import scoring
    from app.services import vad as vad_svc

    settings = get_settings()
    samples = _load_speech(seconds)
    raw = encode(samples, "WAV")

    embedding_svc.warmup(
        settings.embedding_model, settings.model_cache_dir,
        backend=settings.embedding_backend, onnx_threads=settings.onnx_intra_op_threads,
    )

    stages = {k: Timing(k, []) for k in ("decode", "vad", "embed", "score")}
    reference = None

    for _ in range(repeats):
        t0 = time.perf_counter()
        decoded = audio_svc.decode(
            raw, target_rate=settings.target_sample_rate, max_sec=settings.max_audio_sec
        )
        stages["decode"].samples.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        vad_result = vad_svc.apply(
            decoded.samples, decoded.sample_rate,
            threshold=settings.vad_threshold, min_silence_ms=settings.vad_min_silence_ms,
            speech_pad_ms=settings.vad_speech_pad_ms, min_speech_sec=settings.min_speech_sec,
        )
        stages["vad"].samples.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        emb = embedding_svc.extract(
            vad_result.samples, model_name=settings.embedding_model,
            cache_dir=settings.model_cache_dir, backend=settings.embedding_backend,
            onnx_threads=settings.onnx_intra_op_threads,
        )
        stages["embed"].samples.append(time.perf_counter() - t0)

        reference = reference or emb.vector
        t0 = time.perf_counter()
        scoring.match_best(emb.vector, [reference], settings.match_threshold)
        stages["score"].samples.append(time.perf_counter() - t0)

    return {
        k: {"mean_ms": t.mean * 1000, "p95_ms": t.p95 * 1000}
        for k, t in stages.items()
    }


def payload_sizes(seconds: float) -> dict:
    """포맷별 업로드 크기."""
    samples = _load_speech(seconds)
    sizes = {}
    for fmt in ("WAV", "FLAC", "OGG"):
        try:
            sizes[fmt] = len(encode(samples, fmt))
        except Exception:
            sizes[fmt] = None
    return sizes


def concurrency(url: str, user_id: str, audio: bytes, levels: list[int], per_level: int) -> dict:
    """동시 요청 수를 늘려가며 지연·처리량을 잰다.

    CPU 바운드 추론은 스레드를 늘린다고 처리량이 비례해 늘지 않는다. 오히려
    코어를 나눠 쓰느라 개별 지연만 나빠지는 지점이 있고, 그 지점을 알아야
    워커·스레드 수를 정할 수 있다.
    """
    import requests

    def one() -> tuple[float, int]:
        started = time.perf_counter()
        response = requests.post(
            f"{url}/api/v1/verify",
            data={"user_id": user_id},
            files={"file": ("a.wav", audio, "audio/wav")},
            timeout=120,
        )
        return time.perf_counter() - started, response.status_code

    results = {}
    for level in levels:
        # 각 수준마다 워밍업 한 번 — 첫 요청의 캐시 효과를 배제한다
        one()
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=level) as pool:
            outcomes = list(pool.map(lambda _: one(), range(per_level)))
        wall = time.perf_counter() - started

        latencies = [t for t, code in outcomes if code == 200]
        errors = sum(1 for _, code in outcomes if code != 200)
        timing = Timing(f"c{level}", latencies)
        results[level] = {
            "requests": per_level,
            "errors": errors,
            "wall_sec": round(wall, 2),
            "throughput_rps": round(len(latencies) / wall, 2) if wall else 0,
            "p50_ms": round(timing.p50 * 1000, 1),
            "p95_ms": round(timing.p95 * 1000, 1),
        }
        print(f"  동시 {level:>2}: {results[level]['throughput_rps']:>5.2f} req/s  "
              f"p50 {results[level]['p50_ms']:>7.1f}ms  p95 {results[level]['p95_ms']:>7.1f}ms"
              f"{'  오류 ' + str(errors) if errors else ''}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="서버 성능 벤치마크")
    parser.add_argument("--url", default="", help="동시성 측정용 서버 URL (없으면 생략)")
    parser.add_argument("--user-id", default="", help="검증에 쓸 등록된 사용자 ID")
    parser.add_argument("--seconds", type=float, default=5.0, help="테스트 오디오 길이")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--per-level", type=int, default=20)
    args = parser.parse_args()

    print("=" * 68)
    print(f"서버 성능 벤치마크 — 오디오 {args.seconds:.0f}초")
    print("=" * 68)

    print("\n[1] 파이프라인 단계별 (서버 내부, 네트워크 제외)")
    stages = stage_breakdown(args.seconds, args.repeats)
    total = sum(v["mean_ms"] for v in stages.values())
    for name, v in sorted(stages.items(), key=lambda kv: -kv[1]["mean_ms"]):
        share = v["mean_ms"] / total * 100 if total else 0
        bar = "█" * int(share / 3)
        print(f"  {name:<8}{v['mean_ms']:>8.1f}ms (p95 {v['p95_ms']:>6.1f})  {share:>5.1f}% {bar}")
    print(f"  {'합계':<8}{total:>8.1f}ms")

    print("\n[2] 업로드 페이로드 크기")
    sizes = payload_sizes(args.seconds)
    wav = sizes.get("WAV") or 1
    for fmt, size in sizes.items():
        if size is None:
            print(f"  {fmt:<6} 지원 안 함")
            continue
        print(f"  {fmt:<6}{size / 1024:>8.0f}KB  ({size / wav * 100:>5.1f}% of WAV)")

    concurrency_results = {}
    if args.url and args.user_id:
        print(f"\n[3] 동시성 ({args.url})")
        audio = encode(_load_speech(args.seconds), "WAV")
        concurrency_results = concurrency(
            args.url, args.user_id, audio, [1, 2, 4, 8, 16], args.per_level
        )

    report = {
        "audio_seconds": args.seconds,
        "stages": stages,
        "payload_bytes": sizes,
        "concurrency": concurrency_results,
    }
    path = DATA_DIR / "bench.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n보고서: {path}")


if __name__ == "__main__":
    main()
