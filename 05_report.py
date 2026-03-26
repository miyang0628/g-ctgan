"""
05_report.py
------------
실험 결과를 논문용 Table/Figure 형식으로 변환.

출력:
  results/05_table_native.csv      : 조건A 결과표 (기법 고유 비율)
  results/05_table_unified.csv     : 조건B 결과표 (50% 통일) ← 심사자 2 대응
  results/05_auc_comparison.png    : 데이터셋별 AUC 비교 막대그래프
  results/05_ratio_sensitivity.png : 비율 조건 A vs B 비교 (심사자 2 대응)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

RESULTS = Path("results")

METHOD_ORDER = ["none", "smote", "adasyn", "gsmote", "ctgan", "gctgan"]
METHOD_LABELS = {
    "none":   "None",
    "smote":  "SMOTE",
    "adasyn": "ADASYN",
    "gsmote": "G-SMOTE",
    "ctgan":  "CTGAN",
    "gctgan": "G-CTGAN",
}
CLF_ORDER = ["RF", "LGBM", "MLP", "TabNet"]
DS_LABELS = {
    "credit_default": "Credit Default",
    "credit_fraud":   "Credit Fraud",
    "diabetes":       "Pima Diabetes",
    "ibm_attrition":  "IBM Attrition",
}

COLORS = {
    "none":   "#9CA3AF",
    "smote":  "#60A5FA",
    "adasyn": "#34D399",
    "gsmote": "#FBBF24",
    "ctgan":  "#F87171",
    "gctgan": "#6366F1",
}


# ── Table 생성 ────────────────────────────────────────────

def make_table(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """논문 Table 3 형식: 분류기 × 오버샘플링 → AUC/F1/Precision/Recall."""
    sub = df[df["condition"] == condition].copy()
    sub["method"] = pd.Categorical(sub["method"],
                                   categories=METHOD_ORDER, ordered=True)
    sub["classifier"] = pd.Categorical(sub["classifier"],
                                       categories=CLF_ORDER, ordered=True)
    sub = sub.sort_values(["dataset", "classifier", "method"])

    rows = []
    for ds, grp_ds in sub.groupby("dataset", observed=True):
        for clf, grp in grp_ds.groupby("classifier", observed=True):
            for _, row in grp.iterrows():
                rows.append({
                    "Dataset":    DS_LABELS.get(ds, ds),
                    "Classifier": row["classifier"],
                    "Method":     METHOD_LABELS[row["method"]],
                    "AUC":        row["AUC"],
                    "F1":         row["F1"],
                    "Precision":  row["Precision"],
                    "Recall":     row["Recall"],
                })

    return pd.DataFrame(rows)


# ── Figure 1: AUC 비교 막대그래프 ─────────────────────────

def plot_auc_comparison(df: pd.DataFrame):
    """데이터셋 × 분류기별 AUC 비교 (조건 B: unified)."""
    sub = df[df["condition"] == "unified"].copy()
    ds_list  = list(DS_LABELS.keys())
    clf_list = CLF_ORDER

    fig, axes = plt.subplots(len(ds_list), len(clf_list),
                             figsize=(16, 12),
                             sharey=False)

    for i, ds in enumerate(ds_list):
        for j, clf in enumerate(clf_list):
            ax  = axes[i][j]
            grp = sub[(sub["dataset"] == ds) &
                      (sub["classifier"] == clf)]
            if grp.empty:
                ax.set_visible(False)
                continue

            grp = grp.set_index("method").reindex(METHOD_ORDER).dropna()
            methods = [METHOD_LABELS[m] for m in grp.index]
            aucs    = grp["AUC"].values
            colors  = [COLORS[m] for m in grp.index]

            bars = ax.bar(range(len(methods)), aucs,
                          color=colors, width=0.65,
                          edgecolor="white", linewidth=0.5)

            # 최고값 표시
            best_idx = np.argmax(aucs)
            ax.bar(best_idx, aucs[best_idx],
                   color=colors[best_idx],
                   edgecolor="#1e1b4b", linewidth=1.5, width=0.65)

            # AUC 0.75 기준선
            ax.axhline(y=0.75, color="#6B7280",
                       linestyle="--", linewidth=0.7, alpha=0.6)

            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45,
                               ha="right", fontsize=7)
            ax.set_ylim(max(0, min(aucs) - 0.05), 1.0)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_ylabel("AUC" if j == 0 else "", fontsize=8)

            if i == 0:
                ax.set_title(clf, fontsize=9, fontweight="bold")
            if j == 0:
                ax.set_ylabel(DS_LABELS.get(ds, ds) + "\nAUC",
                              fontsize=7)

    plt.suptitle("AUC comparison by dataset, classifier, and oversampling method\n"
                 "(unified 50% minority ratio condition)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = RESULTS / "05_auc_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"AUC 비교 그래프 저장: {out}")


# ── Figure 2: 비율 조건 A vs B 비교 ──────────────────────

def plot_ratio_sensitivity(df: pd.DataFrame):
    """
    심사자 2 지적 대응: 비율 조건에 따른 성능 차이 시각화.
    G-CTGAN에 집중하여 native(42%) vs unified(50%) 비교.
    """
    gctgan = df[df["method"] == "gctgan"].copy()
    ds_list = list(DS_LABELS.keys())

    fig, axes = plt.subplots(1, len(ds_list),
                             figsize=(14, 4), sharey=False)

    for i, ds in enumerate(ds_list):
        ax  = axes[i]
        sub = gctgan[gctgan["dataset"] == ds]

        native  = sub[sub["condition"] == "native"].groupby(
            "classifier")["AUC"].first()
        unified = sub[sub["condition"] == "unified"].groupby(
            "classifier")["AUC"].first()

        x      = np.arange(len(CLF_ORDER))
        w      = 0.35
        native_vals  = [native.get(c, np.nan) for c in CLF_ORDER]
        unified_vals = [unified.get(c, np.nan) for c in CLF_ORDER]

        ax.bar(x - w/2, native_vals,  width=w,
               label="Native ratio", color="#FBBF24",
               edgecolor="white")
        ax.bar(x + w/2, unified_vals, width=w,
               label="Unified 50%",  color="#6366F1",
               edgecolor="white")

        ax.set_title(DS_LABELS.get(ds, ds), fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(CLF_ORDER, fontsize=8)
        ax.set_ylabel("AUC" if i == 0 else "", fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    plt.suptitle("G-CTGAN: sensitivity to minority ratio condition\n"
                 "(native ≈42% vs unified 50%)",
                 fontsize=10, y=1.03)
    plt.tight_layout()
    out = RESULTS / "05_ratio_sensitivity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"비율 민감도 그래프 저장: {out}")


# ── 실행 ─────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(RESULTS / "04_all_results.csv")

    # Table 생성
    t_native  = make_table(df, "native")
    t_unified = make_table(df, "unified")
    t_native.to_csv(RESULTS  / "05_table_native.csv",  index=False)
    t_unified.to_csv(RESULTS / "05_table_unified.csv", index=False)
    print("Table 저장 완료")

    # 하이퍼파라미터 요약 테이블
    hp = pd.read_csv(RESULTS / "04_best_hyperparams.csv")
    hp_summary = (hp[hp["condition"] == "unified"]
                    .groupby(["dataset", "classifier"])
                    ["best_params"].first()
                    .reset_index())
    hp_summary.to_csv(RESULTS / "05_hyperparam_summary.csv",
                      index=False)
    print("하이퍼파라미터 요약 저장: results/05_hyperparam_summary.csv")

    # Figure 생성
    plot_auc_comparison(df)
    plot_ratio_sensitivity(df)

    # 데이터셋별 G-CTGAN 순위 요약 출력
    print("\n[G-CTGAN 순위 요약 (unified, RF+LGBM 기준)]")
    sub = df[(df["condition"] == "unified") &
             (df["classifier"].isin(["RF", "LGBM"]))]
    rank = (sub.groupby(["dataset", "method"])["AUC"]
               .mean()
               .reset_index()
               .sort_values(["dataset", "AUC"], ascending=[True, False]))
    rank["rank"] = rank.groupby("dataset")["AUC"].rank(
        ascending=False).astype(int)
    gctgan_rank = rank[rank["method"] == "gctgan"][
        ["dataset", "AUC", "rank"]]
    print(gctgan_rank.to_string(index=False))