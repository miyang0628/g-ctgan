# G-CTGAN: Oversampling Method Combining GMM and CTGAN

Replication code for:

> Yang, M. (2026). Oversampling Method Combining Gaussian Mixture Model and CTGAN
> for Imbalanced Data Classification.
> *Journal of the Korea Institute of Information and Communication Engineering*, 29(1), 399–406.
> https://doi.org/10.6109/jkiice.2026.29.1.399

---

## Repository structure

```
g-ctgan/
├── data/
│   ├── default of credit card clients.xls
│   ├── creditcard.csv
│   ├── diabetes.csv
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── results/              ← 실행 후 자동 생성
├── data_loader_00.py
├── preprocess_01.py
├── oversample_02.py
├── ablation_gmm_03.py
├── train_evaluate_04.py
├── report_05.py
├── run_all.py
├── experiment.ipynb      ← ✅ 여기서 실행
├── requirements.txt
└── README.md
```

---

## Data download

Place the following files in the `data/` directory:

| File | Source |
|---|---|
| `default of credit card clients.xls` | [UCI ML Repository](https://doi.org/10.24432/C55S3H) |
| `creditcard.csv` | [Kaggle: mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| `diabetes.csv` | [Kaggle: uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| `WA_Fn-UseC_-HR-Employee-Attrition.csv` | [Kaggle: pavansubhasht/ibm-hr-analytics-attrition-dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |

---

## Installation

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
imbalanced-learn>=0.11
ctgan>=0.6
lightgbm>=4.0
matplotlib>=3.7
pytorch-tabnet>=4.1   # optional: TabNet 분류기
openpyxl>=3.1         # for .xls loading
xlrd>=2.0
```

---

## Quick start

```bash
# 전체 파이프라인 일괄 실행
python run_all.py

# 개별 실행
python 00_data_loader.py    # 데이터 로드 및 통계 확인
python 03_ablation_gmm.py   # BIC 곡선 생성 (ablation study)
python 04_train_evaluate.py # 전체 실험 (~수 시간 소요)
python 05_report.py         # 결과 Table/Figure 생성
```

---

## Experimental design

두 가지 소수 범주 비율 조건으로 실험을 수행합니다.

| Condition | Description |
|---|---|
| **Native** | 기법 고유 비율 (G-CTGAN ≈ 42%, SMOTE = 50% 등) |
| **Unified** | 모든 기법 동일하게 50% 통일 — 공정 비교 조건 |

Unified 조건은 심사 과정에서 제기된 "소수 비율 불일치로 인한 직접 비교 제약" 문제에 대응합니다.

---

## Reproducibility

모든 난수 시드는 `RANDOM_STATE = 42`로 고정됩니다.
Train/Test 분할은 `stratify=y`를 적용하여 소수 범주 비율을 보존합니다.
오버샘플링은 훈련 데이터에만 적용하며, 테스트 데이터는 원본 그대로 유지합니다 (data leakage 차단).

---

## Citation

```bibtex
@article{yang2026gctgan,
  title   = {Oversampling Method Combining Gaussian Mixture Model and CTGAN
             for Imbalanced Data Classification},
  author  = {Yang, Munil},
  journal = {Journal of the Korea Institute of Information and Communication Engineering},
  volume  = {29},
  number  = {1},
  pages   = {399--406},
  year    = {2026},
  doi     = {10.6109/jkiice.2026.29.1.399}
}
```
