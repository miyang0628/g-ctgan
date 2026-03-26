"""
run_all.py
----------
전체 파이프라인 순차 실행.
각 단계 소요 시간 출력.
"""

import time
from pathlib import Path


def step(name: str, fn):
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = fn()
    elapsed = time.time() - t0
    print(f"완료: {elapsed:.1f}초")
    return result


if __name__ == "__main__":
    # 결과 폴더 생성
    Path("results").mkdir(exist_ok=True)

    # 00 데이터 로드
    from data_loader_00 import load_all, summarize
    datasets = step("00 데이터 로드", load_all)
    print(summarize(datasets).to_string(index=False))

    # 01 전처리
    from preprocess_01 import preprocess_all
    processed = step("01 전처리 (분할+스케일링)",
                     lambda: preprocess_all(datasets))

    # 03 Ablation (k 선택) — 02보다 먼저 실행하여 optimal_k 확보
    from ablation_gmm_03 import run_ablation
    df_k = step("03 Ablation (GMM BIC k 탐색)",
                lambda: run_ablation(processed))
    optimal_k = dict(zip(df_k["dataset"], df_k["optimal_k"]))
    print(df_k.to_string(index=False))

    # 04 실험 (시간 가장 많이 소요)
    from train_evaluate_04 import run_experiments
    print("\n[주의] CTGAN/G-CTGAN 포함 시 수 시간 소요될 수 있습니다.")
    print("빠른 테스트: 04_train_evaluate.py의 METHODS에서 ctgan/gctgan 제외 후 실행")
    df_res, df_hp = step("04 실험 (전체 분류기 × 오버샘플링)",
                         lambda: run_experiments(processed, optimal_k))

    # 05 리포트
    from report_05 import make_table, plot_auc_comparison, plot_ratio_sensitivity
    import pandas as pd
    df = pd.read_csv("results/04_all_results.csv")

    def make_reports():
        t_native  = make_table(df, "native")
        t_unified = make_table(df, "unified")
        t_native.to_csv("results/05_table_native.csv",   index=False)
        t_unified.to_csv("results/05_table_unified.csv", index=False)
        plot_auc_comparison(df)
        plot_ratio_sensitivity(df)

    step("05 리포트 생성", make_reports)

    print("\n" + "="*60)
    print("전체 파이프라인 완료")
    print("results/ 폴더에서 결과를 확인하세요.")
    print("="*60)