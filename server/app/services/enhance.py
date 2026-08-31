"""DeepFilterNet 기반 음성 향상 (소음 억제).

분리·검증 전에 배경 소음을 억제해 노이즈가 성문 벡터를 오염시키는 것을 차단한다
(02 §2.2). 판별형(discriminative) 모델이라 생성형 향상 모델 대비 화자 특성
왜곡 위험이 낮다는 점이 화자 인증에서 결정적이다 (08 §3).

**향상은 공짜가 아니다.** 이미 깨끗한 오디오에 적용하면 아티팩트를 더해 임베딩
품질을 오히려 떨어뜨릴 수 있다. 채택 여부는 `eval/noise_eval.py`의 실측으로
판단해야 하며, 기본값은 비활성이다.

DeepFilterNet은 48kHz로 동작하므로 16kHz 파이프라인에서는 업/다운샘플링이
따른다. 그 비용도 실측 대상이다.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_df_state = None
_load_lock = threading.Lock()

#: DeepFilterNet이 학습된 샘플레이트
DF_SAMPLE_RATE = 48_000


def _ensure_loaded():
    """향상 모델을 지연 적재한다 (프로세스당 1회)."""
    global _model, _df_state
    if _model is not None:
        return _model, _df_state
    with _load_lock:
        if _model is not None:
            return _model, _df_state
        from df.enhance import init_df

        logger.info("DeepFilterNet 모델 적재 중")
        model, df_state, _ = init_df(log_level="WARNING")
        _model, _df_state = model, df_state
        logger.info("DeepFilterNet 모델 적재 완료")
        return _model, _df_state


def warmup() -> None:
    """모델을 미리 적재해 첫 요청의 콜드 스타트를 없앤다."""
    _ensure_loaded()


def is_loaded() -> bool:
    return _model is not None


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """torchaudio의 고품질 리샘플러를 쓴다.

    audio.py의 선형 보간과 달리 여기서는 정규 경로에서 매번 호출되므로
    (16kHz ↔ 48kHz) 품질이 임베딩에 직접 영향을 준다.
    """
    if src_rate == dst_rate:
        return samples

    import torch
    import torchaudio.functional as AF

    wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
    out = AF.resample(wav, src_rate, dst_rate)
    return out.numpy().astype(np.float32)


def apply(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """소음을 억제한 파형을 반환한다.

    입력 샘플레이트를 그대로 유지해 돌려주므로 호출부는 샘플레이트 변화를
    신경 쓰지 않아도 된다.
    """
    import torch
    from df.enhance import enhance

    model, df_state = _ensure_loaded()

    upsampled = _resample(samples, sample_rate, DF_SAMPLE_RATE)
    wav = torch.from_numpy(upsampled).unsqueeze(0)  # (1, T)

    with torch.no_grad():
        enhanced = enhance(model, df_state, wav)

    out = enhanced.squeeze(0).numpy().astype(np.float32)
    return _resample(out, DF_SAMPLE_RATE, sample_rate)
