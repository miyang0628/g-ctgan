"""
01_preprocess.py
----------------
결측치 처리, 스케일링, train/test 분할.
Data leakage 방지: 스케일러를 train에서만 fit → test에 transform.
stratify=y로 분할하여 소수 범주 비율 유지.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
TEST_SIZE    = 0.30


def preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
    test_size: float  = TEST_SIZE,
) -> dict:
    """
    반환값 dict:
      X_train, X_test : np.ndarray (스케일링 완료)
      y_train, y_test : np.ndarray
      scaler          : 저장/재사용용
      feature_names   : list[str]
    분할 방식: random stratified split (체납자 단위 = 행 단위, 중복 없음).
    """
    feature_names = X.columns.tolist()

    # 1) 수치형만 선택 (범주형은 00에서 이미 인코딩)
    X_arr = X.values.astype(float)

    # 2) 결측치 → 중앙값 대체 (train fit, test transform)
    imputer = SimpleImputer(strategy="median")

    # 3) train/test 분할 (stratify로 소수 비율 보존)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_arr, y.values,
        test_size=test_size,
        random_state=random_state,
        stratify=y.values,
    )

    # 4) Imputer: train에서 fit
    X_tr = imputer.fit_transform(X_tr)
    X_te = imputer.transform(X_te)

    # 5) StandardScaler: train에서 fit → test에 transform
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    return {
        "X_train":       X_tr,
        "X_test":        X_te,
        "y_train":       y_tr,
        "y_test":        y_te,
        "scaler":        scaler,
        "imputer":       imputer,
        "feature_names": feature_names,
    }


def preprocess_all(datasets: dict) -> dict:
    return {name: preprocess(X, y) for name, (X, y) in datasets.items()}


# ── 실행 (단독 테스트용) ─────────────────────────────────

if __name__ == "__main__":
    from data_loader_00 import load_all
    datasets  = load_all()
    processed = preprocess_all(datasets)

    print(f"{'dataset':<20} {'train':>8} {'test':>8} "
          f"{'train_minority%':>16} {'test_minority%':>14}")
    print("-" * 72)
    for name, p in processed.items():
        tr_ratio = p["y_train"].mean() * 100
        te_ratio = p["y_test"].mean()  * 100
        print(f"{name:<20} {len(p['y_train']):>8} {len(p['y_test']):>8} "
              f"{tr_ratio:>15.2f}% {te_ratio:>13.2f}%")