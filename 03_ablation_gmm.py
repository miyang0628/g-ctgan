"""
03_ablation_gmm.py
------------------
GMM 군집 수 k=2~10 탐색 → BIC 최솟값 기준 최적 k 결정.
심사자 2 지적 대응: ablation study 결과를 Figure로 저장.

출력:
  results/03_bic_curves.png   : 데이터셋별 BIC 곡선 (2×2 subplot)
  results/03_optimal_k.csv    : 데이터셋별 최적 k 정리
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from sklearn.mixture import GaussianMixture

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

K_RANGE   = range(2, 11)
RESULTS   = Path("results")
RANDOM_ST = 42


def compute_bic(X_min: np.ndarray, k_range=K_RANGE) -> dict[int, float]:
    """소수 범주 데이터에 대해 k별 BIC 계산."""
    bic = {}
    for k in k_range:
        # 군집 수가 샘플 수보다 클 수 없음
        if k >= len(X_min):
            break
        gm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=RANDOM_ST,
            max_iter=200,
        )
        gm.fit(X_min)
        bic[k] = gm.bic(X_min)
    return bic


def best_k(bic_dict: dict[int, float]) -> int:
    return min(bic_dict, key=bic_dict.get)


def run_ablation(processed: dict) -> pd.DataFrame:
    """모든 데이터셋에 대해 ablation 수행."""
    RESULTS.mkdir(exist_ok=True)
    records = []

    # ── BIC 곡선 그리기 ──────────────────────────────────
    n_ds   = len(processed)
    ncols  = 2
    nrows  = (n_ds + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(10, 4 * nrows))
    axes = axes.flatten()

    ds_names = list(processed.keys())
    labels = {
        "credit_default": "Credit Card Default",
        "credit_fraud":   "Credit Card Fraud",
        "diabetes":       "Pima Diabetes",
        "ibm_attrition":  "IBM HR Attrition",
    }

    for idx, ds_name in enumerate(ds_names):
        p = processed[ds_name]
        X_min = p["X_train"][p["y_train"] == 1]
        print(f"[{ds_name}] 소수 범주 train 샘플 수: {len(X_min)}")

        # 샘플 수가 매우 적으면 k 범위 축소
        k_max = min(10, len(X_min) - 1)
        k_rng = range(2, k_max + 1)

        bic_dict = compute_bic(X_min, k_rng)
        k_opt    = best_k(bic_dict)

        ks   = list(bic_dict.keys())
        bics = list(bic_dict.values())

        ax = axes[idx]
        ax.plot(ks, bics, marker="o", linewidth=1.5,
                color="#2563EB", markersize=5)
        ax.axvline(x=k_opt, color="#DC2626", linestyle="--",
                   linewidth=1.2, label=f"optimal k={k_opt}")
        ax.scatter([k_opt], [bic_dict[k_opt]],
                   color="#DC2626", zorder=5, s=60)
        ax.set_title(labels.get(ds_name, ds_name), fontsize=11)
        ax.set_xlabel("Number of clusters (k)", fontsize=9)
        ax.set_ylabel("BIC", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)

        records.append({
            "dataset":   ds_name,
            "optimal_k": k_opt,
            "min_BIC":   round(bic_dict[k_opt], 2),
            "n_minority_train": len(X_min),
        })

    # 빈 subplot 숨기기
    for j in range(len(ds_names), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("GMM cluster number selection via BIC\n"
                 "(red dashed = optimal k)", fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = RESULTS / "03_bic_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nBIC 곡선 저장: {out_path}")

    df = pd.DataFrame(records)
    df.to_csv(RESULTS / "03_optimal_k.csv", index=False)
    print("최적 k 저장: results/03_optimal_k.csv")
    return df


# ── 실행 ─────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader_00 import load_all
    from preprocess_01  import preprocess_all

    datasets  = load_all()
    processed = preprocess_all(datasets)
    df_k      = run_ablation(processed)

    print("\n[최적 k 요약]")
    print(df_k.to_string(index=False))