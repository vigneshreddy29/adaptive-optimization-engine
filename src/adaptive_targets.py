# =============================================================================
# src/adaptive_targets.py — Dynamic Energy & Carbon Target Setting
# =============================================================================

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_baseline_metrics(df):
    """
    Compute baseline (average) energy and carbon across all batches.
    This is our reference point for adaptive targets.
    """
    baseline = {
        "mean_energy_kwh" : round(df["Est_Total_Energy_kWh"].mean(), 2),
        "mean_carbon_kg"  : round(df["Est_Carbon_kg"].mean(), 4),
        "std_energy_kwh"  : round(df["Est_Total_Energy_kWh"].std(), 2),
        "std_carbon_kg"   : round(df["Est_Carbon_kg"].std(), 4),
        "min_energy_kwh"  : round(df["Est_Total_Energy_kWh"].min(), 2),
        "max_energy_kwh"  : round(df["Est_Total_Energy_kWh"].max(), 2),
        "min_carbon_kg"   : round(df["Est_Carbon_kg"].min(), 4),
        "max_carbon_kg"   : round(df["Est_Carbon_kg"].max(), 4),
    }
    return baseline


def compute_adaptive_targets(df, regulatory_pressure=1.0):
    """
    Compute dynamic energy and carbon targets for the next batch.

    Targets are adaptive — they tighten as performance improves
    and account for regulatory pressure multiplier.

    Args:
        df                  : processed batch DataFrame
        regulatory_pressure : float 1.0=normal, 1.5=high pressure

    Returns:
        dict of adaptive targets
    """
    baseline = compute_baseline_metrics(df)

    # Base reduction targets from config
    energy_reduction = config.ENERGY_REDUCTION_TARGET_PCT * regulatory_pressure
    carbon_reduction = config.CARBON_REDUCTION_TARGET_PCT * regulatory_pressure

    # Clamp to maximum 30% reduction (realistic ceiling)
    energy_reduction = min(energy_reduction, 0.30)
    carbon_reduction = min(carbon_reduction, 0.30)

    # Adaptive target = baseline minus reduction target
    target_energy = baseline["mean_energy_kwh"] * (1 - energy_reduction)
    target_carbon = baseline["mean_carbon_kg"]  * (1 - carbon_reduction)

    # Best achieved so far (floor — we never set target below best)
    best_energy = baseline["min_energy_kwh"]
    best_carbon = baseline["min_carbon_kg"]

    # Final target: between best achieved and reduction target
    final_energy_target = max(target_energy, best_energy * 1.05)
    final_carbon_target = max(target_carbon, best_carbon * 1.05)

    targets = {
        "baseline_energy_kwh"     : baseline["mean_energy_kwh"],
        "baseline_carbon_kg"      : baseline["mean_carbon_kg"],
        "target_energy_kwh"       : round(final_energy_target, 2),
        "target_carbon_kg"        : round(final_carbon_target, 4),
        "energy_reduction_pct"    : round(energy_reduction * 100, 1),
        "carbon_reduction_pct"    : round(carbon_reduction * 100, 1),
        "best_achieved_energy_kwh": baseline["min_energy_kwh"],
        "best_achieved_carbon_kg" : baseline["min_carbon_kg"],
        "regulatory_pressure"     : regulatory_pressure,
    }

    return targets


def evaluate_batch_against_targets(batch_energy, batch_carbon, targets):
    """
    Evaluate a single batch against adaptive targets.

    Returns status and gap analysis.
    """
    energy_gap = batch_energy - targets["target_energy_kwh"]
    carbon_gap = batch_carbon - targets["target_carbon_kg"]

    energy_status = "✅ ON TARGET" if energy_gap <= 0 else "⚠️ ABOVE TARGET"
    carbon_status = "✅ ON TARGET" if carbon_gap <= 0 else "⚠️ ABOVE TARGET"

    energy_gap_pct = (energy_gap / targets["target_energy_kwh"]) * 100
    carbon_gap_pct = (carbon_gap / targets["target_carbon_kg"]) * 100

    return {
        "energy_status"    : energy_status,
        "carbon_status"    : carbon_status,
        "energy_gap_kwh"   : round(energy_gap, 2),
        "carbon_gap_kg"    : round(carbon_gap, 4),
        "energy_gap_pct"   : round(energy_gap_pct, 1),
        "carbon_gap_pct"   : round(carbon_gap_pct, 1),
    }


if __name__ == "__main__":
    df = pd.read_csv(config.PROCESSED_DATA_FILE)
    targets = compute_adaptive_targets(df)

    print("=== ADAPTIVE TARGETS ===")
    for k, v in targets.items():
        print(f"  {k:35s}: {v}")

    print("\n=== BATCH T001 vs TARGETS ===")
    t001 = df[df["Batch_ID"] == "T001"].iloc[0]
    eval_result = evaluate_batch_against_targets(
        t001["Est_Total_Energy_kWh"],
        t001["Est_Carbon_kg"],
        targets
    )
    for k, v in eval_result.items():
        print(f"  {k:25s}: {v}")