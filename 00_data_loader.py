"""
00_data_loader.py
-----------------
4개 공개 데이터셋 로드 및 기본 정보 출력.
각 데이터셋을 (X, y) 형태로 반환하는 함수 제공.

데이터 파일 배치:
  data/
    default of credit card clients.xls   <- UCI
    creditcard.csv                        <- Kaggle ULB
    diabetes.csv                          <- Kaggle Pima
    WA_Fn-UseC_-HR-Employee-Attrition.csv <- Kaggle IBM
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

DATASETS = {
    "credit_default": {
        "desc": "UCI Default of Credit Card Clients",
        "minority_label": 1,
    },
    "credit_fraud": {
        "desc": "Kaggle Credit Card Fraud Detection (ULB)",
        "minority_label": 1,
    },
    "diabetes": {
        "desc": "Kaggle Pima Indians Diabetes",
        "minority_label": 1,
    },
    "ibm_attrition": {
        "desc": "Kaggle IBM HR Analytics Attrition",
        "minority_label": 1,
    },
}


# ── 개별 로더 ─────────────────────────────────────────────

def load_credit_default() -> tuple[pd.DataFrame, pd.Series]:
    """UCI Default of Credit Card Clients (xls or csv)."""
    path = DATA_DIR / "default of credit card clients.xls"
    df = pd.read_excel(path, header=1)
    df.columns = df.columns.str.strip()
    # ID 열 제거
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    target_col = "default payment next month"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


def load_credit_fraud() -> tuple[pd.DataFrame, pd.Series]:
    """Kaggle Credit Card Fraud Detection."""
    path = DATA_DIR / "creditcard.csv"
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    return X, y


def load_diabetes() -> tuple[pd.DataFrame, pd.Series]:
    """Kaggle Pima Indians Diabetes."""
    path = DATA_DIR / "diabetes.csv"
    df = pd.read_csv(path)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"].astype(int)
    return X, y


def load_ibm_attrition() -> tuple[pd.DataFrame, pd.Series]:
    """Kaggle IBM HR Analytics Attrition."""
    path = DATA_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(path)
    # 분산 0인 열 제거 (EmployeeCount, StandardHours 등)
    zero_var = [c for c in df.columns if df[c].nunique() == 1]
    df = df.drop(columns=zero_var)
    # 범주형 → 레이블 인코딩 (전처리는 01에서 세부 처리)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c != "Attrition"]
    for c in cat_cols:
        df[c] = df[c].astype("category").cat.codes
    y = (df["Attrition"] == "Yes").astype(int)
    X = df.drop(columns=["Attrition"])
    return X, y


# ── 통합 로더 ─────────────────────────────────────────────

LOADERS = {
    "credit_default": load_credit_default,
    "credit_fraud":   load_credit_fraud,
    "diabetes":       load_diabetes,
    "ibm_attrition":  load_ibm_attrition,
}


def load_all() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """모든 데이터셋을 로드하여 dict로 반환."""
    result = {}
    for name, fn in LOADERS.items():
        X, y = fn()
        result[name] = (X, y)
    return result


# ── 데이터셋 요약 출력 ────────────────────────────────────

def summarize(datasets: dict) -> pd.DataFrame:
    rows = []
    for name, (X, y) in datasets.items():
        n_total    = len(y)
        n_minority = int(y.sum())
        n_majority = n_total - n_minority
        ratio      = n_minority / n_total * 100
        rows.append({
            "dataset":          name,
            "description":      DATASETS[name]["desc"],
            "n_total":          n_total,
            "n_features":       X.shape[1],
            "n_minority":       n_minority,
            "n_majority":       n_majority,
            "minority_ratio_%": round(ratio, 2),
        })
    return pd.DataFrame(rows)


# ── 실행 ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("데이터셋 로드 시작")
    print("=" * 60)
    datasets = load_all()
    summary  = summarize(datasets)
    print(summary.to_string(index=False))
    out = Path("results")
    out.mkdir(exist_ok=True)
    summary.to_csv(out / "00_dataset_summary.csv", index=False)
    print("\n결과 저장: results/00_dataset_summary.csv")