# korean-voc-laya-finetune

한국어 카드 상담을 8개 업무 그룹으로 분류하도록 Laya를 파인튜닝하고,
같은 test set에서 Jev와 비교하는 프로젝트입니다. 범위는 [goal.md](goal.md)를 참고하세요.

## 준비

Python 3.11 이상과 학습용 GPU를 권장합니다. 공개 레시피 기준 16GB GPU에서는
작은 micro batch와 gradient checkpointing이 필요합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Laya 데이터 생성

기본 입력은 상담사 답변에 의한 정답 누출을 줄이기 위해 `text_customer`입니다.

```bash
python scripts/prepare_laya_data.py
```

빈 고객 발화 한 건을 제외하고 계속하려면 `--skip-empty`를 사용합니다.

```bash
python scripts/prepare_laya_data.py --skip-empty
```

상담 종료 후 분류 실험에는 `--input-column text`를 지정할 수 있습니다. 학습과 평가는
항상 같은 입력 열을 사용해야 합니다.

## 2. 파인튜닝

기본 베이스 체크포인트는 `2nugu/laya-ko`입니다.

```bash
python scripts/train_laya.py \
  --model 2nugu/laya-ko \
  --epochs 4 \
  --micro-batch 4 \
  --grad-accum 16
```

validation macro-F1이 가장 높은 체크포인트만 `models/laya-card-groups/`에 저장하며,
temperature도 validation logits으로 보정합니다. CUDA 메모리가 부족하면
`--micro-batch 1 --grad-accum 64`로 낮춥니다.

## 3. 고정 test set 평가

```bash
python scripts/evaluate_laya.py
```

결과는 `results/laya/metrics.json`과 `results/laya/predictions.jsonl`에 저장됩니다.
Jev도 동일한 입력 열, 그룹 순서와 설명을 사용하고 같은 prediction schema로 저장해야
공정하게 비교할 수 있습니다.

## 현재까지의 실험 결과

### 데이터와 환경

- 분류 대상: 8개 상담 그룹
- 입력: 고객 발화만 포함한 `text_customer`
- 학습/검증/테스트: 5,175 / 573 / 773건
- test 원본 774건 중 고객 발화가 비어 있는 1건 제외
- 베이스 모델: `2nugu/laya-ko`
- 학습 장비: Apple M4 16GB, PyTorch MPS
- 목적 함수: RLCD proper-reward 항 + cross-entropy
- 체크포인트 선택 기준: validation Macro-F1

### 1차 학습: epoch 1–4

| Epoch | Validation Accuracy | Validation Macro-F1 |
| ---: | ---: | ---: |
| 1 | 0.6213 | 0.5156 |
| 2 | 0.6475 | 0.5573 |
| 3 | 0.6684 | 0.5872 |
| 4 | **0.6771** | **0.6149** |

1차 학습에서는 epoch 4가 최적이었다. 해당 체크포인트의 test 결과는 다음과 같다.

| 지표 | Test |
| --- | ---: |
| Accuracy | 0.6779 |
| Macro-F1 | 0.6210 |
| ECE | 0.0985 |
| Brier score | 0.4789 |
| Log loss | 1.0863 |
| MPS batch 추론 속도 | 54.5 ms/건 |

### 추가 학습: epoch 5–8

epoch 4 체크포인트에서 encoder/head learning rate를 각각 `3e-6`, `3e-5`로 낮춰
4 epoch를 추가 학습했다.

| Epoch | Validation Accuracy | Validation Macro-F1 |
| ---: | ---: | ---: |
| 5 | **0.6876** | **0.6297** |
| 6 | 0.6806 | 0.6239 |
| 7 | 0.6771 | 0.6123 |
| 8 | 0.6806 | 0.6195 |

추가 학습 구간에서는 epoch 5가 최적이었으며, epoch 6부터 validation 성능이 하락해
plateau 또는 과적합 신호가 나타났다.

| 지표 | Epoch 4 | Epoch 5 |
| --- | ---: | ---: |
| Test Accuracy | **0.6779** | 0.6753 |
| Test Macro-F1 | 0.6210 | **0.6222** |
| ECE | **0.0985** | 0.1230 |
| Brier score | **0.4789** | 0.4936 |
| Log loss | **1.0863** | 1.1216 |

epoch 5는 validation Macro-F1을 개선했지만 test Macro-F1 개선은 약 0.12%p에 그쳤고,
Accuracy와 calibration은 소폭 나빠졌다. 따라서 Jev와의 주 비교에는 validation 기준으로
선택한 epoch 5를 사용하되, calibration 관점에서는 epoch 4 결과도 함께 보고한다.

### Epoch 5 그룹별 test 성능

| 그룹 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| 결제/출금 | 0.730 | 0.848 | 0.785 |
| 이용내역/매출 | 0.629 | 0.566 | 0.596 |
| 한도/심사 | 0.815 | 0.671 | 0.736 |
| 대출/리볼빙 | 0.509 | 0.500 | 0.505 |
| 카드관리/증명 | 0.580 | 0.784 | 0.667 |
| 혜택/부가서비스 | 0.583 | 0.500 | 0.538 |
| 생활/공공요금 | 0.722 | 0.553 | 0.627 |
| 이용방법 | 0.640 | 0.444 | 0.525 |

모델 체크포인트, 원본·변환 데이터, 개별 예측 및 평가 산출물은 저장소에 커밋하지 않는다.
동일한 결과는 위 절차로 로컬에서 재생성할 수 있다.

## 다음 단계

Jev에 동일한 773개 test 입력과 동일한 8개 선택지 설명을 제공하고 Accuracy,
Macro-F1, 그룹별 성능, ECE/Brier score, 지연시간 및 비용을 비교한다.

## 출처

학습 루프는 Laya의 공개 RLCD 파인튜닝 노트북과 `2nugu/laya-ko-decision-onnx`의
한국어 학습 레시피를 현재 데이터셋에 맞게 수정한 것입니다.

- https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
- https://github.com/2nugu/laya-ko-decision-onnx
