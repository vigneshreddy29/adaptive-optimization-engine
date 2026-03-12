# =============================================================================
# src/optimization_engine.py — Multi-Objective Optimization Engine
# =============================================================================

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_processed_data():
    """Load the processed batch features dataset."""
    df = pd.read_csv(config.PROCESSED_DATA_FILE)
    print(f"[Optimizer] Loaded {len(df)} batches from processed data.")
    return df


def normalize_scores(df):
    """
    Normalize all objective scores to 0-1 range.
    For energy and carbon — lower is better, so we invert.
    """
    df = df.copy()

    def minmax(series, invert=False):
        min_v = series.min()
        max_v = series.max()
        if max_v == min_v:
            return pd.Series([0.5] * len(series), index=series.index)
        result = (series - min_v) / (max_v - min_v)
        return (1 - result) if invert else result

    df["Norm_Quality"]     = minmax(df["Quality_Score"])
    df["Norm_Yield"]       = minmax(df["Yield_Score"])
    df["Norm_Performance"] = minmax(df["Performance_Score"])
    df["Norm_Energy"]      = minmax(df["Est_Total_Energy_kWh"], invert=True)
    df["Norm_Carbon"]      = minmax(df["Est_Carbon_kg"], invert=True)

    return df


def compute_composite_score(df, weights):
    """
    Compute weighted composite score for each batch.

    Args:
        df      : DataFrame with normalized objective columns
        weights : dict with keys w_quality, w_yield, w_performance,
                  w_energy, w_carbon

    Returns:
        Series of composite scores (0-1)
    """
    score = (
        weights["w_quality"]     * df["Norm_Quality"]     +
        weights["w_yield"]       * df["Norm_Yield"]       +
        weights["w_performance"] * df["Norm_Performance"] +
        weights["w_energy"]      * df["Norm_Energy"]      +
        weights["w_carbon"]      * df["Norm_Carbon"]
    )
    return score


def identify_pareto_front(df):
    """
    Identify Pareto-optimal batches using vectorized numpy.

    A batch is Pareto-optimal if no other batch dominates it
    on ALL objectives simultaneously.

    Works on normalized columns if available, otherwise normalizes
    raw scores inline. O(n²) loop replaced with numpy broadcasting.

    Objectives to MAXIMIZE:
        Quality_Score, Yield_Score, Performance_Score,
        Norm_Energy (inverted — lower energy = better),
        Norm_Carbon (inverted — lower carbon = better)
    """
    df = df.copy()

    # Use normalized columns if already computed, else normalize inline
    if "Norm_Quality" in df.columns:
        score_matrix = df[[
            "Norm_Quality", "Norm_Yield", "Norm_Performance",
            "Norm_Energy",  "Norm_Carbon"
        ]].values
    else:
        # Inline min-max normalization (invert energy and carbon)
        def minmax(series, invert=False):
            mn, mx = series.min(), series.max()
            if mx == mn:
                return np.ones(len(series)) * 0.5
            result = (series - mn) / (mx - mn)
            return (1 - result) if invert else result

        score_matrix = np.column_stack([
            minmax(df["Quality_Score"]),
            minmax(df["Yield_Score"]),
            minmax(df["Performance_Score"]),
            minmax(df["Est_Total_Energy_kWh"], invert=True),
            minmax(df["Est_Carbon_kg"],        invert=True),
        ])

    # Vectorized Pareto dominance check using numpy broadcasting
    # scores shape: (n, m) — n batches, m objectives
    # For each batch i, check if ANY other batch j dominates it:
    #   j dominates i iff: all(scores[j] >= scores[i])
    #                  and any(scores[j] >  scores[i])
    n = len(score_matrix)

    # Expand dims for broadcasting: (n, 1, m) vs (1, n, m)
    s_i = score_matrix[:, np.newaxis, :]   # shape (n, 1, m)
    s_j = score_matrix[np.newaxis, :, :]   # shape (1, n, m)

    # j >= i on all objectives: shape (n, n, m) → (n, n)
    all_geq  = np.all(s_j >= s_i, axis=2)
    # j > i on at least one objective: shape (n, n, m) → (n, n)
    any_gt   = np.any(s_j >  s_i, axis=2)

    # j dominates i: shape (n, n)
    dominates = all_geq & any_gt

    # Remove self-domination diagonal
    np.fill_diagonal(dominates, False)

    # Batch i is Pareto-optimal if NO batch j dominates it
    is_pareto = ~np.any(dominates, axis=1)

    df["Is_Pareto"] = is_pareto
    pareto_df = df[is_pareto].copy()

    return df, pareto_df


def run_optimization():
    """
    Run the full multi-objective optimization across all scenarios.

    Returns:
        results : dict with scenario name -> best batch info
        full_df : DataFrame with all scores and Pareto flags
    """
    print("=" * 60)
    print("  MULTI-OBJECTIVE OPTIMIZATION ENGINE")
    print("=" * 60)

    df = load_processed_data()
    df = normalize_scores(df)

    # Identify Pareto front
    df, pareto_df = identify_pareto_front(df)
    print(f"\n[Pareto] {len(pareto_df)} Pareto-optimal batches identified "
          f"out of {len(df)} total.")
    print(f"[Pareto] Pareto batch IDs: {pareto_df['Batch_ID'].tolist()}")

    results = {}

    print("\n[Scenarios] Evaluating all 3 Golden Signature scenarios...\n")

    for scenario_key, scenario_cfg in config.OPTIMIZATION_SCENARIOS.items():
        weights = {
            "w_quality"    : scenario_cfg["w_quality"],
            "w_yield"      : scenario_cfg["w_yield"],
            "w_performance": scenario_cfg["w_performance"],
            "w_energy"     : scenario_cfg["w_energy"],
            "w_carbon"     : scenario_cfg["w_carbon"],
        }

        df[f"Score_{scenario_key}"] = compute_composite_score(df, weights)

        # Best batch = highest composite score
        best_idx  = df[f"Score_{scenario_key}"].idxmax()
        best_batch = df.loc[best_idx]

        results[scenario_key] = {
            "scenario_label"   : scenario_cfg["label"],
            "best_batch_id"    : best_batch["Batch_ID"],
            "composite_score"  : round(best_batch[f"Score_{scenario_key}"], 4),
            "quality_score"    : round(best_batch["Quality_Score"], 2),
            "yield_score"      : round(best_batch["Yield_Score"], 2),
            "performance_score": round(best_batch["Performance_Score"], 2),
            "energy_kwh"       : round(best_batch["Est_Total_Energy_kWh"], 2),
            "carbon_kg"        : round(best_batch["Est_Carbon_kg"], 4),
            "is_pareto"        : bool(best_batch["Is_Pareto"]),
            "process_params"   : {
                col: round(float(best_batch[col]), 3)
                for col in config.PROCESS_PARAM_COLS
            },
        }

        print(f"  Scenario : {scenario_cfg['label']}")
        print(f"  Best Batch  : {results[scenario_key]['best_batch_id']}")
        print(f"  Score       : {results[scenario_key]['composite_score']}")
        print(f"  Quality     : {results[scenario_key]['quality_score']}")
        print(f"  Yield       : {results[scenario_key]['yield_score']}")
        print(f"  Energy(kWh) : {results[scenario_key]['energy_kwh']}")
        print(f"  Carbon(kg)  : {results[scenario_key]['carbon_kg']}")
        print(f"  Pareto      : {results[scenario_key]['is_pareto']}")
        print("-" * 40)

    print("\n[Optimizer] Complete.")
    return results, df


if __name__ == "__main__":
    results, full_df = run_optimization()