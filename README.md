# 신용카드 사기 거래 탐지 — 라벨 30건으로 하는 이상탐지

[Dacon 신용카드 사기 거래 탐지 AI 경진대회](https://dacon.io/competitions/official/235930/overview/description) 개인 프로젝트

**최종 결과: Public 0.9305 / Private 0.9054 (macro-F1)** — 첫 제출(0.6614 / 0.6396)에서 9번의 제출을 거쳐 개선

## 문제

- train 113,842건은 **라벨 없음**, 라벨은 val 28,462건뿐이고 그중 사기는 **30건(0.105%)**
- 피처 `V1~V30`은 익명화된 수치형 변수 → 도메인 지식 없이 데이터 구조만으로 사기를 찾아야 함
- 평가지표 macro-F1: "전부 정상" 예측도 accuracy 99.9%가 나오므로 accuracy는 쓸 수 없음

자동화나 속도보다 **원리를 이해하며 단계적으로 쌓아가는 것**을 목표로 Phase를 나눠 진행했습니다. 지표(혼동행렬·F1·macro-F1)와 마할라노비스 거리는 numpy로 직접 구현한 뒤 sklearn/scipy와 값이 같은지 검증했습니다.

## 핵심 발견

1. **데이터 정체 파악** — 세 분할을 합친 284,807행이 Kaggle "Credit Card Fraud Detection"과 일치. `V29`·`V30`은 RobustScaler를 거친 금액·시간이며 역변환하면 센트·초 단위 정수로 정확히 복원됨
2. **"평면" 구조** — 거래의 88.4%가 30차원 공간에서 정확히 한 평면 위(분산 0인 방향 2개)에 놓여 있고, **val 사기의 83%(25/30)는 평면 밖**에 있음. 라벨 없이 train만으로 찾은 구조라 누수가 없음
3. **Elliptic Envelope 실패 원인 규명** — 처음엔 "알고리즘 한계"로 기록했지만, 실제 원인은 MCD가 평면 위 점만으로 부분집합을 채워 공분산이 특이행렬이 되는 것(exact fit). `support_fraction`을 0.994로 올리자 평면에 얇은 두께가 생기며 최고 성능 모델이 됨
4. **공유 코드 ablation** — 공개된 오토인코더(Public 0.926)를 한 요소씩 바꿔보니 성능의 핵심은 구조가 아니라 **L1 손실**(MSE로 바꾸면 0.91 → 0.51). MSE는 극단값까지 복원하려 해서 사기도 잘 복원해버림
5. **검증 지표 선택** — 양성 30건에서는 val F1(사기 2~3건 차이)보다 **PR-AUC가 리더보드 순서를 더 잘 맞춤**을 세 번 확인. ROC-AUC는 불균형에서 성능을 과대평가함 (LOF: ROC-AUC 0.75, PR-AUC 0.007)

> 결론: 이 데이터의 핵심은 모델 종류보다 **극단값에 휘둘리지 않는 정상 기준(강건 추정)** 이었습니다.
> 마할라노비스(0.66) → L1 오토인코더(0.91) → 트리밍한 MCD(0.92)

## 리더보드 이력

| 제출 | 방법 | Public | Private |
|---|---|---|---|
| 1 | 마할라노비스 거리 (Phase 2) | 0.6614 | 0.6396 |
| 2 | Isolation Forest (Dacon 베이스라인 설정) | 0.6861 | 0.6877 |
| 3 | 평면 이탈 거리 + IF (순위 평균) | 0.8207 | 0.8145 |
| 4 | 평면 이탈 거리 단독 | 0.8294 | 0.8306 |
| 5 | PCA(k=28) 재구성 오차 | 0.8658 | 0.8464 |
| 6 | L1 오토인코더 + 코사인 유사도 | 0.9091 | 0.8971 |
| 7 | Elliptic Envelope (support_fraction 0.994) | 0.9248 | **0.9105** |
| **8** | **EE 1차 공분산 + threshold 0.12% (최종)** | **0.9305** | 0.9054 |
| 9 | Normalizer → EE | 0.9122 | 0.8971 |

## 진행 단계

| Phase | 노트북 | 내용 |
|---|---|---|
| 1 | [01_EDA](notebooks/01_EDA.ipynb) | 분포 비교(KS-test), 데이터셋 정체 확인, 중복·불균형 점검 |
| 2 | [02_baseline_statistical](notebooks/02_baseline_statistical.ipynb) | 평가지표·마할라노비스 직접 구현, threshold 스윕, 첫 제출 |
| 3 | [03_classical_anomaly_models](notebooks/03_classical_anomaly_models.ipynb) | Isolation Forest · One-Class SVM · LOF · Elliptic Envelope 비교 |
| 3b | [03b_feature_engineering](notebooks/03b_feature_engineering.ipynb) | 금액·시간 복원, 평면 구조 발견, EE 실패 원인 규명 |
| 3c | [03c_model_revalidation](notebooks/03c_model_revalidation.ipynb) | 새 피처로 LOF · OCSVM · kNN 재검증, LOF 실패 원인 분석 |
| 4 | [04_autoencoder](notebooks/04_autoencoder.ipynb) | PCA 재구성 오차, 오토인코더 |
| 4b | [04b_shared_ae_ablation](notebooks/04b_shared_ae_ablation.ipynb) | 공개 코드 재현과 요소별 ablation |
| 4c | [04c_my_elliptic_envelope](notebooks/04c_my_elliptic_envelope.ipynb) | EE 재도전, 하이퍼파라미터 점검 → 최종 모델 |
| 5 | [05_supervised_hybrid](notebooks/05_supervised_hybrid.ipynb) | 로지스틱 회귀 · LightGBM · 하이브리드를 반복 교차검증으로 비교 |

Phase 5에서는 val 라벨로 지도학습을 시도했지만, 반복 층화 5겹 교차검증(10회, 50조각)에서 미리 정한 기준(이긴 조각 70% 이상)을 넘는 방법이 없어 **비지도 EE를 최종 모델로 확정**했습니다. 단계별 상세 기록은 [진행상황_업데이트.md](진행상황_업데이트.md)에 있습니다.

## 최종 모델

```
원본 V1~V30
  → MinCovDet(support_fraction=0.994)의 1차(raw) 공분산으로 마할라노비스 거리
  → threshold = train 점수 상위 0.12%
```

남은 한계: 평면 밖 사기 25/25 탐지, **평면 위 사기 0/5**, 오탐 5건 — 평면 위 사기는 지도학습으로도 신호를 찾지 못했습니다.

## 실행 방법

대회 데이터는 용량과 배포 조건 때문에 저장소에 포함하지 않았습니다. Dacon 대회 페이지에서 받아 `data/`에 넣으면 노트북을 순서대로 실행할 수 있습니다.

```
data/train.csv, data/val.csv, data/test.csv, data/sample_submission.csv
```

주요 패키지: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, torch, lightgbm, imbalanced-learn
