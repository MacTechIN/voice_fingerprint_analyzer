"""동시 요청 안전성 테스트 (Phase 5).

**실제로 서버를 무너뜨린 버그를 회귀 방지한다.** silero-vad의 TorchScript 모델은
내부에 RNN 상태를 들고 있어, 한 인스턴스를 여러 스레드가 동시에 쓰면 상태가
깨진다. 실측에서 동시 요청 4건 중 2건이 500으로 떨어졌다.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import soundfile as sf

from app.services import vad as vad_svc

from .conftest import SAMPLE_RATE, synth_speech, unique_user

VAD_KWARGS = dict(threshold=0.5, min_silence_ms=300, speech_pad_ms=30, min_speech_sec=1.5)


def _wav(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestVadThreadSafety:
    """VAD 모델의 스레드 격리."""

    def test_concurrent_vad_does_not_fail(self):
        """여러 스레드가 동시에 VAD를 호출해도 예외가 나지 않는다.

        수정 전에는 TorchScript 내부 상태가 깨져 RuntimeError가 났다.
        """
        speech = synth_speech(3.0)

        def run() -> float:
            result = vad_svc.apply(speech, SAMPLE_RATE, **VAD_KWARGS)
            return result.speech_duration_sec

        with ThreadPoolExecutor(max_workers=8) as pool:
            durations = list(pool.map(lambda _: run(), range(24)))

        assert len(durations) == 24
        assert all(d > 0 for d in durations)

    def test_concurrent_results_are_identical(self):
        """같은 입력은 스레드와 무관하게 같은 결과를 낸다.

        상태가 섞이면 예외 없이 **조용히 다른 결과**가 나올 수도 있다. 그쪽이
        더 위험하므로 값까지 확인한다.
        """
        speech = synth_speech(3.0, f0=130.0, seed=5)
        expected = vad_svc.apply(speech, SAMPLE_RATE, **VAD_KWARGS).speech_duration_sec

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda _: vad_svc.apply(speech, SAMPLE_RATE, **VAD_KWARGS).speech_duration_sec,
                    range(16),
                )
            )

        for value in results:
            assert value == pytest.approx(expected, abs=1e-6)

    def test_each_thread_gets_its_own_model(self):
        """스레드마다 별도 인스턴스를 쓴다."""
        seen: list[int] = []

        def capture() -> None:
            vad_svc.apply(synth_speech(3.0), SAMPLE_RATE, **VAD_KWARGS)
            seen.append(id(vad_svc._ensure_loaded()))

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: capture(), range(4)))

        assert len(set(seen)) > 1, "모든 스레드가 같은 인스턴스를 공유하고 있다"

    def test_torch_threads_preserved_under_concurrency(self):
        """동시 호출에서도 전역 torch 스레드 수가 복원된다."""
        import torch

        before = torch.get_num_threads()

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(
                pool.map(
                    lambda _: vad_svc.apply(synth_speech(3.0), SAMPLE_RATE, **VAD_KWARGS),
                    range(8),
                )
            )

        assert torch.get_num_threads() == before


class TestConcurrentVerify:
    """API 수준 동시 요청."""

    def test_concurrent_verify_all_succeed(self, client, fresh_storage, speech_wav_bytes):
        """동시 검증 요청이 모두 성공한다 — 수정 전에는 절반이 500이었다."""
        user = unique_user()
        client.post(
            "/api/v1/enroll",
            data={"user_id": user},
            files={"file": ("a.wav", speech_wav_bytes, "audio/wav")},
        )

        def verify() -> int:
            return client.post(
                "/api/v1/verify",
                data={"user_id": user},
                files={"file": ("a.wav", speech_wav_bytes, "audio/wav")},
            ).status_code

        with ThreadPoolExecutor(max_workers=6) as pool:
            codes = list(pool.map(lambda _: verify(), range(12)))

        assert codes == [200] * 12, f"동시 요청 중 실패: {codes}"

    def test_concurrent_verify_gives_consistent_scores(
        self, client, fresh_storage, speech_wav_bytes
    ):
        """동시 요청이 같은 판정 점수를 낸다 — 상태 오염은 조용히 값을 바꾼다."""
        user = unique_user()
        client.post(
            "/api/v1/enroll",
            data={"user_id": user},
            files={"file": ("a.wav", speech_wav_bytes, "audio/wav")},
        )

        def score() -> float:
            body = client.post(
                "/api/v1/verify",
                data={"user_id": user},
                files={"file": ("a.wav", speech_wav_bytes, "audio/wav")},
            ).json()
            return body["raw_cosine"]

        with ThreadPoolExecutor(max_workers=6) as pool:
            scores = list(pool.map(lambda _: score(), range(12)))

        assert max(scores) - min(scores) < 1e-4, f"점수가 흔들린다: {scores}"
