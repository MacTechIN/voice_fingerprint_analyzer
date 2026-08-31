# 평가 하네스 — 임계값 캘리브레이션 (Phase 6)

임계값을 관례값으로 두면 시스템이 실제로 어느 지점에서 실패하는지 알 수 없다.
이 하네스는 **실제 화자 데이터**로 EER·minDCF를 측정해 운영 임계값의 근거를 만든다.

## 데이터

[LibriSpeech](https://www.openslr.org/12/) (CC BY 4.0). 화자 ID가 붙어 있어
Genuine/Impostor 트라이얼을 구성할 수 있다.

| split | 화자 | 용도 |
| :--- | ---: | :--- |
| `dev-clean` | 40 | 평가 트라이얼 |
| `test-clean` | 40 | AS-Norm 임포스터 코호트 |

**두 split의 화자는 겹치지 않는다.** 코호트에 평가 화자가 섞이면 정규화가 자기
자신을 참조하게 되어 EER이 실제보다 좋게 나온다 — 측정 자체가 무의미해진다.

### 준비

```bash
cd server && mkdir -p .data && cd .data
curl -O https://www.openslr.org/resources/12/dev-clean.tar.gz
curl -O https://www.openslr.org/resources/12/test-clean.tar.gz
tar xzf dev-clean.tar.gz && tar xzf test-clean.tar.gz
```

`.data/`는 `.gitignore` 대상이다 (약 706MB).

## 실행

```bash
# 1. 캘리브레이션 — EER·minDCF 측정, 임계값 산출
.venv/bin/python -m eval.calibrate

# 2. 코호트 DB 적재 (운영 서버가 읽는다)
VG_DATABASE_URL="postgresql://voiceguard:voiceguard@127.0.0.1:54321/voiceguard" \
  .venv/bin/python -m eval.seed_cohort --replace
```

임베딩은 `.data/cache/*.npz`에 캐시된다. 수천 발화의 임베딩을 매 실험마다 다시
뽑으면 임계값 몇 개 바꿔보는 것조차 수십 분이 걸린다. 모델을 바꿨다면 캐시를
지우거나 `force=True`로 재추출한다.

## 구성

| 모듈 | 역할 |
| :--- | :--- |
| `dataset.py` | 발화 로드, Genuine/Impostor 트라이얼 생성 |
| `extract.py` | 임베딩 일괄 추출 + npz 캐시 |
| `metrics.py` | DET 곡선, EER, minDCF, 점수 분포 통계 |
| `calibrate.py` | 원시 코사인 vs AS-Norm 비교, 임계값 산출 |
| `seed_cohort.py` | 임포스터 코호트 DB 적재 |
| `noise_eval.py` | 음성 향상(DeepFilterNet) 효과를 SNR별로 측정 |

### 백엔드별 캐시 분리

임베딩 캐시와 보고서는 백엔드·모델별로 분리된다
(`.data/cache/<backend>__<model>/`, `.data/calibration_report__<backend>__<model>.json`).
분리하지 않으면 모델을 바꿔도 이전 임베딩을 재사용해 **측정이 조용히 틀린다** —
오류가 나지 않아 눈치채기 어려운 종류의 실수다.

```bash
# 백엔드를 바꿔 캘리브레이션
VG_EMBEDDING_BACKEND=speechbrain \
VG_EMBEDDING_MODEL=speechbrain/spkrec-ecapa-voxceleb \
VG_EMBEDDING_DIM=192 \
  .venv/bin/python -m eval.calibrate
```

### 음성 향상 효과 측정

```bash
.venv/bin/python -m eval.noise_eval --max-per-speaker 4 --max-speakers 20
```

깨끗한 조건과 SNR 20/10/5/0dB 각각에서, 향상 적용 전후의 EER을 비교한다.
**"향상을 켜면 좋아진다"를 가정하지 않는다** — 이미 깨끗한 오디오에 적용하면
아티팩트를 더해 오히려 나빠질 수 있으므로 채택 여부를 데이터로 정한다.

등록 발화는 항상 깨끗한 것을 쓰고 검증 발화에만 소음을 섞는다. 실제 서비스가
그렇기 때문이다 — 등록은 통제된 환경에서 한 번, 검증은 아무 데서나 한다.

> 주입하는 잡음은 **가우시안 백색잡음**이다. 광대역 정상 잡음은 소음 억제가 가장
> 잘 다루는 종류이므로, 여기서 나온 개선폭은 **낙관적인 상한**으로 읽어야 한다.
> 실제 배경 대화·음악·잔향에서는 더 어렵다. MUSAN 같은 실환경 소음 코퍼스로
> 재측정하는 것이 다음 단계다.

## 측정 방법론

**평가는 운영 경로와 같은 조건이어야 한다.** `extract.py`는 서버와 동일하게
VAD를 적용하고, VAD가 반려했을 발화는 트라이얼에서도 제외한다. 전처리를 건너뛰고
측정하면 실제 운영 성능과 다른 숫자가 나온다.

**트라이얼은 Genuine/Impostor 수를 맞춘다.** 한쪽이 많으면 EER이 그쪽 분포에
치우쳐 계산된다.

**지표**

- **EER** — FAR과 FRR이 같아지는 지점의 오류율. 두 오류를 대칭적으로 다룰 때의
  기준이며, 그 지점의 임계값이 운영 임계값 후보다.
- **minDCF** — 탐지 비용의 최소값 (`p_target=0.01`, `C_miss=C_fa=1`). 사칭 시도가
  드문 실제 환경을 반영해 FAR에 더 큰 가중을 둔다.
- **분리도** — Genuine/Impostor 평균 간격을 표준편차로 나눈 값. EER과 함께 보면
  성능 변화의 원인을 읽기 쉽다.

## 결과

`.data/calibration_report.json`에 전체 수치가 남는다. 요약은
[06_Development_Plan.md](../../docs/06_Development_Plan.md)의 Phase 6 절 참조.

## 한계

- **LibriSpeech는 낭독 음성이다.** 조용한 환경에서 또렷하게 읽은 영어 오디오북이며,
  실제 서비스의 잡음·짧은 발화·다국어 조건과 다르다. 여기서 나온 임계값은
  **출발점이지 최종값이 아니다.** 운영 데이터가 쌓이면 그 분포로 재캘리브레이션해야 한다.
- **화자 80명**은 통계적으로 넉넉하지 않다. 임계값 추정치에 표본 오차가 있다.
- 코호트는 LibriSpeech 화자로 구성되어 있어, 실사용자 집단과 음향 특성이 다르면
  정규화 효과가 측정값보다 작을 수 있다.
