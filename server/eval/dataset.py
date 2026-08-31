"""LibriSpeech 기반 평가 데이터 준비.

평가 화자와 코호트 화자를 **반드시 분리한다**. AS-Norm 코호트에 평가 화자가
섞이면 정규화가 자기 자신을 참조하게 되어 EER이 실제보다 좋게 나온다 —
측정 자체가 무의미해진다 (02 §4.2 "실제 사용자와 무관한 화자").

기본 배치:
    dev-clean  (40화자) → 평가 트라이얼
    test-clean (40화자) → 임포스터 코호트
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / ".data" / "LibriSpeech"


@dataclass(frozen=True)
class Utterance:
    """발화 하나."""

    path: Path
    speaker: str

    @property
    def key(self) -> str:
        return self.path.stem


def load_utterances(split: str, *, max_per_speaker: int = 8, seed: int = 0) -> list[Utterance]:
    """해당 split에서 화자별로 최대 N개 발화를 고른다.

    화자당 개수를 제한하는 이유는 임베딩 추출 비용 때문이며, 화자별로 같은 수를
    뽑아 특정 화자가 트라이얼을 지배하지 않게 한다.
    """
    root = DATA_ROOT / split
    if not root.exists():
        raise FileNotFoundError(
            f"{root} 가 없습니다. eval/README.md의 데이터 준비 절차를 먼저 실행하세요."
        )

    rng = random.Random(seed)
    out: list[Utterance] = []
    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        flacs = sorted(speaker_dir.rglob("*.flac"))
        if len(flacs) > max_per_speaker:
            flacs = rng.sample(flacs, max_per_speaker)
        out.extend(Utterance(path=p, speaker=speaker_dir.name) for p in sorted(flacs))
    return out


@dataclass(frozen=True)
class Trial:
    """검증 시도 한 건 — 등록 발화와 검증 발화의 쌍."""

    enroll: str
    test: str
    is_same_speaker: bool


def build_trials(
    utterances: list[Utterance],
    *,
    genuine_per_speaker: int = 20,
    impostor_per_speaker: int = 20,
    seed: int = 0,
) -> list[Trial]:
    """Genuine(동일 화자) / Impostor(다른 화자) 트라이얼을 만든다.

    두 종류의 개수를 맞춰 EER이 한쪽 분포에 치우쳐 계산되지 않게 한다.
    """
    rng = random.Random(seed)
    by_speaker: dict[str, list[Utterance]] = {}
    for u in utterances:
        by_speaker.setdefault(u.speaker, []).append(u)

    speakers = sorted(by_speaker)
    trials: list[Trial] = []

    for spk in speakers:
        own = by_speaker[spk]
        if len(own) < 2:
            continue

        # Genuine: 같은 화자의 서로 다른 발화 쌍
        for _ in range(genuine_per_speaker):
            a, b = rng.sample(own, 2)
            trials.append(Trial(enroll=a.key, test=b.key, is_same_speaker=True))

        # Impostor: 다른 화자의 발화
        others = [s for s in speakers if s != spk]
        for _ in range(impostor_per_speaker):
            a = rng.choice(own)
            b = rng.choice(by_speaker[rng.choice(others)])
            trials.append(Trial(enroll=a.key, test=b.key, is_same_speaker=False))

    rng.shuffle(trials)
    return trials
