"""화자 임베딩 추출 — 교체 가능한 백엔드.

두 백엔드를 지원한다.

* `speechbrain` — ECAPA-TDNN (192차원). 착수가 빠르고 문서가 좋다.
* `wespeaker`   — ResNet34-LM 등 ONNX (256차원). AS-Norm·서빙이 완결된
  프로덕션 지향 툴킷의 모델이며, onnxruntime로 실행하므로 PyTorch 추론보다
  가볍다 (02 §3.1, 07 §2).

**백엔드를 바꾸면 임베딩 차원과 잠재 공간이 모두 달라진다.** 기존 벡터와
호환되지 않으므로 재등록이 필요하고, 그 판단 근거는 벡터와 함께 저장된
`model` 메타데이터뿐이다. 그래서 임베딩에는 반드시 모델 식별자를 실어 보낸다.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

_backend = None
_backend_key: tuple | None = None
_load_lock = threading.Lock()


@dataclass(frozen=True)
class SpeakerEmbedding:
    """화자 임베딩과 그 출처 메타데이터."""

    vector: list[float]
    dim: int
    model: str
    l2_normalized: bool


class _SpeechBrainBackend:
    """SpeechBrain ECAPA-TDNN (192차원)."""

    def __init__(self, model_name: str, cache_dir: str) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        logger.info("SpeechBrain 임베딩 모델 적재 중: %s", model_name)
        self.model_name = model_name
        self._classifier = EncoderClassifier.from_hparams(
            source=model_name,
            savedir=f"{cache_dir}/{model_name.replace('/', '__')}",
            run_opts={"device": "cpu"},
        )
        logger.info("SpeechBrain 임베딩 모델 적재 완료")

    def encode(self, samples: np.ndarray) -> np.ndarray:
        wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            # 반환 shape: (batch, 1, dim)
            emb = self._classifier.encode_batch(wav).squeeze(0).squeeze(0)
        return emb.numpy()


class _WeSpeakerOnnxBackend:
    """WeSpeaker ONNX 모델 (ResNet34-LM 등).

    입력은 80차원 Kaldi fbank이며, 모델 내부에 정규화가 없으므로 학습 때와 동일한
    평균 차감(CMN)을 여기서 해야 한다. 이를 빠뜨리면 임베딩이 조용히 나빠진다 —
    오류가 나지 않아 눈에 띄지 않는 종류의 실수다.
    """

    #: HF 저장소 → ONNX 파일명
    KNOWN_MODELS = {
        "Wespeaker/wespeaker-voxceleb-resnet34-LM": "voxceleb_resnet34_LM.onnx",
        "Wespeaker/wespeaker-voxceleb-resnet34": "voxceleb_resnet34.onnx",
        "Wespeaker/wespeaker-voxceleb-campplus-LM": "voxceleb_CAM++_LM.onnx",
        "Wespeaker/wespeaker-voxceleb-ecapa-tdnn512": "voxceleb_ECAPA512.onnx",
        "Wespeaker/wespeaker-ecapa-tdnn512-LM": "voxceleb_ECAPA512_LM.onnx",
        "Wespeaker/wespeaker-voxceleb-resnet221-LM": "voxceleb_resnet221_LM.onnx",
        "Wespeaker/wespeaker-voxceleb-resnet293-LM": "voxceleb_resnet293_LM.onnx",
    }

    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        *,
        sample_rate: int = 16_000,
        intra_op_threads: int = 4,
    ) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        filename = self.KNOWN_MODELS.get(model_name)
        if filename is None:
            raise ValueError(
                f"알 수 없는 WeSpeaker 모델: {model_name}. "
                f"지원 목록: {sorted(self.KNOWN_MODELS)}"
            )

        logger.info("WeSpeaker ONNX 모델 적재 중: %s", model_name)
        path = hf_hub_download(model_name, filename, cache_dir=f"{cache_dir}/hf")
        self.model_name = model_name
        self.sample_rate = sample_rate
        # 스레드 수는 지연 시간과 동시 처리량의 트레이드오프다. 실측(5.9초 오디오,
        # ResNet34-LM): 1스레드 206ms / 2스레드 108ms / 4스레드 58ms / 8스레드 32ms.
        # 단일 요청 지연을 줄이려면 늘리고, 동시 요청이 많으면 줄여 코어를 요청 간에
        # 나눠 쓰는 편이 낫다. FastAPI가 이미 요청별 워커 스레드를 쓰기 때문이다.
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info("WeSpeaker ONNX 모델 적재 완료 (%s)", filename)

    def _fbank(self, samples: np.ndarray) -> np.ndarray:
        """WeSpeaker 학습 설정과 동일한 80차원 Kaldi fbank + CMN."""
        import torchaudio.compliance.kaldi as kaldi

        # kaldi.fbank는 int16 스케일(±32768)을 전제로 한다. float32 [-1,1]을 그대로
        # 넣으면 에너지가 달라져 특징이 어긋난다.
        wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
        wav = wav * (1 << 15)

        feats = kaldi.fbank(
            wav,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,  # 추론에서는 재현성을 위해 끈다 (학습 시에는 1.0)
            sample_frequency=self.sample_rate,
            window_type="hamming",
            use_energy=False,
        )
        # CMN — 발화별 평균 차감. 채널·녹음 조건 차이를 흡수한다.
        feats = feats - feats.mean(dim=0, keepdim=True)
        return feats.numpy()

    def encode(self, samples: np.ndarray) -> np.ndarray:
        feats = self._fbank(samples)[None, :, :]  # (1, T, 80)
        embs = self._session.run(None, {self._input_name: feats.astype(np.float32)})[0]
        return embs[0]


def _make_backend(backend: str, model_name: str, cache_dir: str, *, onnx_threads: int = 4):
    if backend == "speechbrain":
        return _SpeechBrainBackend(model_name, cache_dir)
    if backend == "wespeaker":
        return _WeSpeakerOnnxBackend(model_name, cache_dir, intra_op_threads=onnx_threads)
    raise ValueError(f"알 수 없는 임베딩 백엔드: {backend} (speechbrain | wespeaker)")


def _ensure_loaded(backend: str, model_name: str, cache_dir: str, onnx_threads: int = 4):
    """임베딩 백엔드를 지연 적재한다 (설정 조합당 1회)."""
    global _backend, _backend_key
    key = (backend, model_name)
    if _backend is not None and _backend_key == key:
        return _backend
    with _load_lock:
        if _backend is not None and _backend_key == key:
            return _backend
        _backend = _make_backend(backend, model_name, cache_dir, onnx_threads=onnx_threads)
        _backend_key = key
        return _backend


def warmup(
    model_name: str, cache_dir: str, backend: str = "speechbrain", onnx_threads: int = 4
) -> None:
    """모델을 미리 적재해 첫 요청의 콜드 스타트를 없앤다."""
    _ensure_loaded(backend, model_name, cache_dir, onnx_threads)


def is_loaded() -> bool:
    """모델이 적재되어 있는지 — `/health`에 노출한다."""
    return _backend is not None


def reset() -> None:
    """적재된 백엔드를 해제한다 (테스트에서 백엔드를 바꿀 때 사용)."""
    global _backend, _backend_key
    _backend = None
    _backend_key = None


def extract(
    samples: np.ndarray,
    *,
    model_name: str,
    cache_dir: str,
    backend: str = "speechbrain",
    onnx_threads: int = 4,
    l2_normalize: bool = True,
) -> SpeakerEmbedding:
    """발화 파형에서 화자 임베딩을 추출한다.

    Args:
        samples: VAD를 통과한 16kHz mono float32 파형.
        l2_normalize: 코사인 유사도는 방향만 보므로 저장 단계에서 미리
            정규화해 두면 이후 내적만으로 유사도를 얻는다.
    """
    impl = _ensure_loaded(backend, model_name, cache_dir, onnx_threads)
    vec = np.asarray(impl.encode(samples), dtype=np.float32).reshape(-1)

    if l2_normalize:
        norm = float(np.linalg.norm(vec))
        # 완전 무음이면 norm이 0이 될 수 있다. VAD가 앞단에서 걸러주지만,
        # 0으로 나누어 NaN을 DB에 저장하는 사고를 막는다.
        if norm > 0:
            vec = vec / norm

    values = vec.tolist()
    return SpeakerEmbedding(
        vector=values,
        dim=len(values),
        model=model_name,
        l2_normalized=l2_normalize,
    )
