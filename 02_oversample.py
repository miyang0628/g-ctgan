"""
02_oversample.py
----------------
6개 오버샘플링 기법 × 2개 비율 조건 → 12개 훈련셋 생성.

조건 A (native)  : 기법 고유 비율 (SMOTE=50%, G-CTGAN≈42% 등 — 원 논문 재현)
조건 B (unified) : 모든 기법 소수 범주 50% 통일       ← 심사자 2 지적 대응

G-CTGAN 알고리즘:
  1. GMM BIC 최적 k 결정 (03_ablation 모듈에서 k 주입)
  2. 각 군집에 CTGAN 독립 적용
  3. 생성된 샘플을 원본 훈련셋에 병합

의존 패키지:
  pip install imbalanced-learn ctgan scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from imblearn.over_sampling import SMOTE, ADASYN
from ctgan import CTGAN


# ────────────────────────────────────────────────────────
# G-SMOTE (GMM + SMOTE)
# ────────────────────────────────────────────────────────

def _gmm_split(X_min: np.ndarray, k: int, random_state: int = 42):
    """소수 범주를 GMM으로 k개 군집으로 분리."""
    gm = GaussianMixture(n_components=k, random_state=random_state)
    labels = gm.fit_predict(X_min)
    clusters = [X_min[labels == i] for i in range(k)]
    return clusters, labels


def gsmote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_gmm: int,
    target_ratio: float = 0.5,
    random_state: int   = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """G-SMOTE: GMM 군집화 후 각 군집에 SMOTE 적용."""
    X_min = X_train[y_train == 1]
    X_maj = X_train[y_train == 0]
    n_maj = len(X_maj)
    n_min_target = int(n_maj * target_ratio / (1 - target_ratio))
    n_need = max(0, n_min_target - len(X_min))

    if n_need == 0:
        return X_train, y_train

    clusters, _ = _gmm_split(X_min, k_gmm, random_state)
    # 각 군집 기여 비율 (군집 크기에 비례)
    sizes   = np.array([len(c) for c in clusters])
    weights = sizes / sizes.sum()

    synthetic_all = []
    for i, cluster in enumerate(clusters):
        n_gen = int(n_need * weights[i])
        if len(cluster) < 2 or n_gen == 0:
            continue
        # SMOTE 적용 (군집 내부에서만)
        dummy_y = np.ones(len(cluster), dtype=int)
        # k_neighbors는 군집 크기에 맞게 조정
        k_nn = min(3, len(cluster) - 1)
        sm = SMOTE(
            sampling_strategy={1: len(cluster) + n_gen},
            k_neighbors=k_nn,
            random_state=random_state,
        )
        try:
            X_res, _ = sm.fit_resample(
                np.vstack([cluster, X_maj[:len(cluster)]]),
                np.hstack([dummy_y, np.zeros(len(cluster), dtype=int)]),
            )
            # 생성된 소수 범주 샘플만 추출
            new_samples = X_res[len(cluster):len(cluster) + n_gen]
            synthetic_all.append(new_samples)
        except Exception:
            continue

    if synthetic_all:
        X_syn = np.vstack(synthetic_all)
        X_new = np.vstack([X_train, X_syn])
        y_new = np.hstack([y_train, np.ones(len(X_syn), dtype=int)])
        return X_new, y_new
    return X_train, y_train


# ────────────────────────────────────────────────────────
# G-CTGAN (GMM + CTGAN) — 제안 기법
# ────────────────────────────────────────────────────────

def gctgan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_gmm: int,
    target_ratio: float  = 0.50,
    epochs: int          = 300,
    random_state: int    = 42,
    verbose: bool        = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    G-CTGAN: GMM 군집화 후 각 군집에 CTGAN 독립 적용.
    target_ratio: 최종 소수 범주 비율 목표 (0.5 = 50%)
    """
    X_min = X_train[y_train == 1]
    X_maj = X_train[y_train == 0]
    n_maj = len(X_maj)
    n_min_target = int(n_maj * target_ratio / (1 - target_ratio))
    n_need = max(0, n_min_target - len(X_min))

    if n_need == 0:
        return X_train, y_train

    clusters, _ = _gmm_split(X_min, k_gmm, random_state)
    sizes   = np.array([len(c) for c in clusters])
    weights = sizes / sizes.sum()

    n_cols = X_train.shape[1]
    col_names = [f"f{i}" for i in range(n_cols)]
    synthetic_all = []

    for i, cluster in enumerate(clusters):
        n_gen = int(n_need * weights[i])
        if len(cluster) < 10 or n_gen == 0:
            # 군집이 너무 작으면 단순 복제로 대체
            if n_gen > 0:
                idx = np.random.RandomState(random_state).randint(
                    0, len(cluster), size=n_gen)
                synthetic_all.append(cluster[idx])
            continue

        df_cluster = pd.DataFrame(cluster, columns=col_names)
        ctgan = CTGAN(epochs=epochs, verbose=verbose)
        ctgan.fit(df_cluster)
        syn_df = ctgan.sample(n_gen)
        synthetic_all.append(syn_df.values)

    if synthetic_all:
        X_syn = np.vstack(synthetic_all)
        X_new = np.vstack([X_train, X_syn])
        y_new = np.hstack([y_train, np.ones(len(X_syn), dtype=int)])
        return X_new, y_new
    return X_train, y_train


# ────────────────────────────────────────────────────────
# CTGAN (단독, 군집 없음)
# ────────────────────────────────────────────────────────

def ctgan_only(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_ratio: float = 0.50,
    epochs: int         = 300,
    verbose: bool       = False,
) -> tuple[np.ndarray, np.ndarray]:
    X_min = X_train[y_train == 1]
    X_maj = X_train[y_train == 0]
    n_maj = len(X_maj)
    n_min_target = int(n_maj * target_ratio / (1 - target_ratio))
    n_need = max(0, n_min_target - len(X_min))

    if n_need == 0 or len(X_min) < 10:
        return X_train, y_train

    col_names = [f"f{i}" for i in range(X_min.shape[1])]
    df_min = pd.DataFrame(X_min, columns=col_names)
    model  = CTGAN(epochs=epochs, verbose=verbose)
    model.fit(df_min)
    syn_df = model.sample(n_need)
    X_syn  = syn_df.values
    return (np.vstack([X_train, X_syn]),
            np.hstack([y_train, np.ones(len(X_syn), dtype=int)]))


# ────────────────────────────────────────────────────────
# 통합 오버샘플링 함수
# ────────────────────────────────────────────────────────

def apply_oversampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    k_gmm: int          = 5,
    target_ratio: float = 0.50,
    random_state: int   = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    method: 'none' | 'smote' | 'adasyn' | 'gsmote' | 'ctgan' | 'gctgan'
    target_ratio: 소수 범주 목표 비율 (조건 B에서 0.5 고정)
    """
    sr = {1: int(len(y_train[y_train == 0]) * target_ratio / (1 - target_ratio))}

    if method == "none":
        return X_train, y_train

    elif method == "smote":
        sm = SMOTE(sampling_strategy=sr, k_neighbors=3,
                   random_state=random_state)
        return sm.fit_resample(X_train, y_train)

    elif method == "adasyn":
        ad = ADASYN(sampling_strategy=target_ratio,
                    random_state=random_state)
        try:
            return ad.fit_resample(X_train, y_train)
        except Exception:
            # ADASYN은 극심한 불균형 시 실패 가능 → SMOTE fallback
            sm = SMOTE(sampling_strategy=sr, k_neighbors=3,
                       random_state=random_state)
            return sm.fit_resample(X_train, y_train)

    elif method == "gsmote":
        return gsmote(X_train, y_train, k_gmm=k_gmm,
                      target_ratio=target_ratio,
                      random_state=random_state)

    elif method == "ctgan":
        return ctgan_only(X_train, y_train,
                          target_ratio=target_ratio)

    elif method == "gctgan":
        return gctgan(X_train, y_train, k_gmm=k_gmm,
                      target_ratio=target_ratio)

    else:
        raise ValueError(f"알 수 없는 기법: {method}")


# ── 실행 ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from data_loader_00  import load_all
    from preprocess_01   import preprocess_all

    datasets  = load_all()
    processed = preprocess_all(datasets)

    METHODS = ["none", "smote", "adasyn", "gsmote", "ctgan", "gctgan"]
    CONDITIONS = {"native": None, "unified": 0.50}

    for ds_name, p in list(processed.items())[:1]:  # 테스트: 1개만
        print(f"\n[{ds_name}]")
        for method in METHODS[:3]:  # 빠른 기법만 테스트
            X_os, y_os = apply_oversampling(
                p["X_train"], p["y_train"], method=method)
            ratio = y_os.mean() * 100
            print(f"  {method:<10}: total={len(y_os):>7}, "
                  f"minority={y_os.sum():>6}, ratio={ratio:.1f}%")