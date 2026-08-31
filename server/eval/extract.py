"""평가용 임베딩 일괄 추출.

추출 결과를 npz로 캐시한다. 5천여 발화의 임베딩을 매 실험마다 다시 뽑으면
임계값을 몇 개 바꿔보는 것조차 수십 분이 걸린다.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from app.config import get_settings
from app.services import embedding as embedding_svc
from app.services import vad as vad_svc
from eval.dataset import Utterance

logger = logging.getLogger(__name__)


def _load_audio(path: Path, target_rate: int) -> np.ndarray:
    """flac을 16kHz 모노 float32로 읽는다."""
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]
    if rate != target_rate:
        duration = len(mono) / rate
        dst_len = int(round(duration * target_rate))
        idx = np.linspace(0, len(mono) - 1, num=dst_len)
        mono = np.interp(idx, np.arange(len(mono)), mono).astype(np.float32)
    return np.ascontiguousarray(mono, dtype=np.float32)


def extract_all(
    utterances: list[Utterance],
    cache_path: Path,
    *,
    apply_vad: bool = True,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """발화별 임베딩을 추출하고 npz로 캐시한다.

    Args:
        apply_vad: 서버 파이프라인과 동일하게 VAD를 적용할지 여부. 평가가 실제
            운영 경로와 같은 조건이어야 측정값이 의미를 갖는다.
    """
    if cache_path.exists() and not force:
        logger.info("캐시된 임베딩 사용: %s", cache_path)
        data = np.load(cache_path)
        return {k: data[k] for k in data.files}

    settings = get_settings()
    embedding_svc.warmup(settings.embedding_model, settings.model_cache_dir)

    out: dict[str, np.ndarray] = {}
    skipped = 0
    started = time.perf_counter()

    for i, utt in enumerate(utterances, 1):
        samples = _load_audio(utt.path, settings.target_sample_rate)

        if apply_vad:
            try:
                result = vad_svc.apply(
                    samples,
                    settings.target_sample_rate,
                    threshold=settings.vad_threshold,
                    min_silence_ms=settings.vad_min_silence_ms,
                    speech_pad_ms=settings.vad_speech_pad_ms,
                    min_speech_sec=settings.min_speech_sec,
                )
                samples = result.samples
            except Exception:
                # 서버라면 반려했을 발화다. 평가에서도 동일하게 제외한다.
                skipped += 1
                continue

        emb = embedding_svc.extract(
            samples, model_name=settings.embedding_model, cache_dir=settings.model_cache_dir
        )
        out[utt.key] = np.asarray(emb.vector, dtype=np.float32)

        if i % 200 == 0:
            rate = i / (time.perf_counter() - started)
            logger.info("임베딩 추출 %d/%d (%.1f 발화/초)", i, len(utterances), rate)

    logger.info(
        "임베딩 추출 완료: %d개 (VAD 반려 %d개), %.1f초",
        len(out),
        skipped,
        time.perf_counter() - started,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **out)
    return out
