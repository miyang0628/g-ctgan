"""
04_train_evaluate.py
--------------------
4개 분류기 × 6개 오버샘플링 기법 × 2개 비율 조건 × 4개 데이터셋 실험.
GridSearchCV로 최적 하이퍼파라미터 탐색 (심사자 2 지적 대응).

출력:
  results/04_all_results.csv       : 전체 실험 결과
  results/04_best_hyperparams.csv  : 분류기별 최적 하이퍼파라미터
"""

import importlib
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score,
                              precision_score, recall_score)
import lightgbm as lgb

warnings.filterwarnings("ignore")
RESULTS = Path("results")
RANDOM_ST = 42
CV_FOLDS  = 3   # GridSearch CV fold 수 (시간 절약, 3 권장)

# ── 하이퍼파라미터 탐색 공간 ─────────────────────────────

PARAM_GRIDS = {
    "RF": {
        "n_estimators":    [100, 200],
        "max_depth":       [None, 10, 20],
        "min_samples_leaf":[1, 3],
        "class_weight":    ["balanced"],
    },
    "LGBM": {
        "n_estimators":    [100, 200],
        "learning_rate":   [0.05, 0.1],
        "num_leaves":      [31, 63],
        "class_weight":    ["balanced"],
    },
    "MLP": {
        "hidden_layer_sizes": [(64, 32), (128, 64)],
        "alpha":              [1e-4, 1e-3],
        "max_iter":           [300],
    },
}

# TabNet은 GridSearch 대신 고정 파라미터 사용 (학습 비용 고려)
TABNET_PARAMS = {
    "n_d":           16,
    "n_a":           16,
    "n_steps":       3,
    "gamma":         1.3,
    "n_independent": 2,
    "n_shared":      2,
    "momentum":      0.02,
    "mask_type":     "entmax",
}


# ── 분류기 팩토리 ────────────────────────────────────────

def get_base_model(clf_name: str):
    if clf_name == "RF":
        return RandomForestClassifier(random_state=RANDOM_ST,
                                      n_jobs=-1)
    elif clf_name == "LGBM":
        return lgb.LGBMClassifier(random_state=RANDOM_ST,
                                   verbose=-1, n_jobs=-1)
    elif clf_name == "MLP":
        return MLPClassifier(random_state=RANDOM_ST,
                             early_stopping=True,
                             validation_fraction=0.1)
    else:
        raise ValueError(f"Unknown: {clf_name}")


def grid_search(clf_name: str, X_tr: np.ndarray,
                y_tr: np.ndarray) -> tuple:
    """GridSearchCV 실행 → (best_estimator, best_params)"""
    base  = get_base_model(clf_name)
    grid  = PARAM_GRIDS[clf_name]
    cv    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                            random_state=RANDOM_ST)
    gs    = GridSearchCV(base, grid, cv=cv,
                         scoring="roc_auc", n_jobs=-1,
                         refit=True, verbose=0)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_


def train_tabnet(X_tr: np.ndarray, y_tr: np.ndarray):
    """TabNet 학습 (pytorch_tabnet 필요)."""
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        import torch
        clf = TabNetClassifier(**TABNET_PARAMS,
                               seed=RANDOM_ST,
                               verbose=0)
        clf.fit(
            X_tr, y_tr,
            eval_metric=["auc"],
            max_epochs=100,
            patience=15,
            batch_size=1024,
            virtual_batch_size=128,
        )
        return clf, TABNET_PARAMS
    except ImportError:
        print("  [경고] pytorch_tabnet 미설치 → TabNet 건너뜀")
        return None, {}


# ── 평가 함수 ────────────────────────────────────────────

def evaluate(clf, clf_name: str,
             X_te: np.ndarray, y_te: np.ndarray) -> dict:
    if clf_name == "TabNet":
        prob = clf.predict_proba(X_te)[:, 1]
    else:
        prob = clf.predict_proba(X_te)[:, 1]

    pred = (prob >= 0.5).astype(int)
    return {
        "AUC":       round(roc_auc_score(y_te, prob), 4),
        "F1":        round(f1_score(y_te, pred, zero_division=0), 4),
        "Precision": round(precision_score(y_te, pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, pred, zero_division=0), 4),
    }


# ── 메인 실험 루프 ────────────────────────────────────────

def run_experiments(processed: dict, optimal_k: dict) -> pd.DataFrame:
    """
    processed  : {ds_name: {X_train, X_test, y_train, y_test}}
    optimal_k  : {ds_name: k}  (03_ablation 결과)
    """
    # 이렇게 교체
    oversample = importlib.import_module("02_oversample")
    apply_oversampling = oversample.apply_oversampling

    RESULTS.mkdir(exist_ok=True)
    METHODS    = ["none", "smote", "adasyn", "gsmote", "ctgan", "gctgan"]
    CLF_NAMES  = ["RF", "LGBM", "MLP", "TabNet"]
    CONDITIONS = {"native": None, "unified": 0.50}

    # 기법별 고유 비율 (native 조건, 원 논문 설정 재현)
    NATIVE_RATIOS = {
        "none":   None,
        "smote":  0.50,
        "adasyn": 0.50,
        "gsmote": 0.50,
        "ctgan":  0.486,
        "gctgan": 0.421,
    }

    all_records   = []
    best_hp_records = []

    for ds_name, p in processed.items():
        k = optimal_k.get(ds_name, 5)
        print(f"\n{'='*60}")
        print(f"데이터셋: {ds_name}  (optimal k={k})")
        print(f"{'='*60}")

        for cond_name, fixed_ratio in CONDITIONS.items():
            print(f"\n  [비율 조건: {cond_name}]")

            for method in METHODS:
                ratio = fixed_ratio if fixed_ratio else \
                        NATIVE_RATIOS.get(method, 0.50)
                print(f"    오버샘플링: {method:<10} ratio={ratio}")

                try:
                    X_os, y_os = apply_oversampling(
                        p["X_train"], p["y_train"],
                        method=method,
                        k_gmm=k,
                        target_ratio=ratio if ratio else \
                                     NATIVE_RATIOS[method],
                    )
                except Exception as e:
                    print(f"      [오류] {e} — 건너뜀")
                    continue

                min_ratio_actual = y_os.mean() * 100

                for clf_name in CLF_NAMES:
                    print(f"      분류기: {clf_name}", end=" ... ")
                    try:
                        if clf_name == "TabNet":
                            clf, hp = train_tabnet(X_os, y_os)
                            if clf is None:
                                continue
                        else:
                            clf, hp = grid_search(clf_name,
                                                  X_os, y_os)
                        metrics = evaluate(clf, clf_name,
                                           p["X_test"],
                                           p["y_test"])
                        print(f"AUC={metrics['AUC']}")
                        record = {
                            "dataset":    ds_name,
                            "condition":  cond_name,
                            "method":     method,
                            "classifier": clf_name,
                            "minority_%": round(min_ratio_actual, 1),
                            **metrics,
                        }
                        all_records.append(record)
                        best_hp_records.append({
                            "dataset":    ds_name,
                            "condition":  cond_name,
                            "method":     method,
                            "classifier": clf_name,
                            "best_params": str(hp),
                        })
                    except Exception as e:
                        print(f"[오류] {e}")
                        continue

    df_res = pd.DataFrame(all_records)
    df_hp  = pd.DataFrame(best_hp_records)
    df_res.to_csv(RESULTS / "04_all_results.csv", index=False)
    df_hp.to_csv(RESULTS / "04_best_hyperparams.csv", index=False)
    print("\n결과 저장 완료")
    return df_res, df_hp


# ── 실행 ─────────────────────────────────────────────────

if __name__ == "__main__":
    data_loader  = importlib.import_module("00_data_loader")
    preprocess   = importlib.import_module("01_preprocess")
    ablation_gmm = importlib.import_module("03_ablation_gmm")

    load_all       = data_loader.load_all
    preprocess_all = preprocess.preprocess_all
    run_ablation   = ablation_gmm.run_ablation

    datasets  = load_all()
    processed = preprocess_all(datasets)
    df_k      = run_ablation(processed)
    optimal_k = dict(zip(df_k["dataset"], df_k["optimal_k"]))

    df_res, df_hp = run_experiments(processed, optimal_k)
    print(f"\n총 실험 수: {len(df_res)}")
    print(df_res.groupby(["dataset", "method"])["AUC"]
                .max().reset_index().to_string(index=False))