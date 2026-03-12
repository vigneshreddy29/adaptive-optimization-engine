# =============================================================================
# src/recommendation.py — Actionable Recommendation Generator
# =============================================================================

import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.golden_signature import load_golden_signatures
from src.adaptive_targets import compute_adaptive_targets, evaluate_batch_against_targets


def generate_recommendation(scenario_key, current_batch_params=None):
    """
    Generate actionable recommendations by comparing current batch
    parameters against the Golden Signature for a given scenario.

    Args:
        scenario_key         : which GS scenario to compare against
        current_batch_params : dict of current process parameters
                               (if None, uses fleet average as baseline)

    Returns:
        recommendation dict with parameter adjustments and expected impact
    """
    gs_store = load_golden_signatures()
    df       = pd.read_csv(config.PROCESSED_DATA_FILE)
    targets  = compute_adaptive_targets(df)

    if scenario_key not in gs_store:
        return {"error": f"Scenario '{scenario_key}' not found."}

    gs = gs_store[scenario_key]
    gs_params = gs["process_params"]

    # Use fleet average if no current batch provided
    if current_batch_params is None:
        current_batch_params = {
            col: round(float(df[col].mean()), 3)
            for col in config.PROCESS_PARAM_COLS
        }

    # Compute parameter adjustments
    adjustments = {}
    for param in config.PROCESS_PARAM_COLS:
        current_val = current_batch_params.get(param, 0)
        gs_val      = gs_params.get(param, 0)
        delta       = gs_val - current_val
        delta_pct   = (delta / current_val * 100) if current_val != 0 else 0

        if abs(delta_pct) < 1.0:
            direction = "MAINTAIN"
        elif delta > 0:
            direction = "INCREASE"
        else:
            direction = "DECREASE"

        adjustments[param] = {
            "current"    : round(current_val, 3),
            "recommended": round(gs_val, 3),
            "delta"      : round(delta, 3),
            "delta_pct"  : round(delta_pct, 1),
            "direction"  : direction,
        }

    # Expected impact vs current average
    current_energy = float(df["Est_Total_Energy_kWh"].mean())
    current_carbon = float(df["Est_Carbon_kg"].mean())
    gs_energy      = gs["energy_kwh"]
    gs_carbon      = gs["carbon_kg"]

    energy_saving_pct = (current_energy - gs_energy) / current_energy * 100
    carbon_saving_pct = (current_carbon - gs_carbon) / current_carbon * 100
    quality_improvement = gs["quality_score"] - float(df["Quality_Score"].mean())
    yield_improvement   = gs["yield_score"]   - float(df["Yield_Score"].mean())

    # Evaluate GS against adaptive targets
    target_eval = evaluate_batch_against_targets(
        gs_energy, gs_carbon, targets
    )

    recommendation = {
        "scenario_key"        : scenario_key,
        "scenario_label"      : gs["scenario_label"],
        "golden_batch_id"     : gs["source_batch_id"],
        "gs_version"          : gs["version"],
        "parameter_adjustments": adjustments,
        "expected_impact": {
            "energy_saving_pct"    : round(energy_saving_pct, 1),
            "carbon_saving_pct"    : round(carbon_saving_pct, 1),
            "quality_improvement"  : round(quality_improvement, 2),
            "yield_improvement"    : round(yield_improvement, 2),
            "target_energy_kwh"    : targets["target_energy_kwh"],
            "target_carbon_kg"     : targets["target_carbon_kg"],
        },
        "target_evaluation"   : target_eval,
        "gs_composite_score"  : gs["composite_score"],
    }

    return recommendation


def print_recommendation(rec):
    """Pretty-print a recommendation to console."""
    print(f"\n{'='*60}")
    print(f"  RECOMMENDATION — {rec['scenario_label']}")
    print(f"  Golden Batch: {rec['golden_batch_id']} "
          f"(v{rec['gs_version']}, score: {rec['gs_composite_score']})")
    print(f"{'='*60}")

    print("\n  PARAMETER ADJUSTMENTS:")
    for param, adj in rec["parameter_adjustments"].items():
        arrow = {"INCREASE": "↑", "DECREASE": "↓", "MAINTAIN": "→"}.get(
            adj["direction"], "→")
        print(f"    {param:25s}: {adj['current']:8.2f} {arrow} "
              f"{adj['recommended']:8.2f}  ({adj['delta_pct']:+.1f}%)")

    print("\n  EXPECTED IMPACT vs CURRENT AVERAGE:")
    ei = rec["expected_impact"]
    print(f"    Energy Saving    : {ei['energy_saving_pct']:+.1f}%")
    print(f"    Carbon Saving    : {ei['carbon_saving_pct']:+.1f}%")
    print(f"    Quality Change   : {ei['quality_improvement']:+.2f} points")
    print(f"    Yield Change     : {ei['yield_improvement']:+.2f} points")

    print("\n  TARGET EVALUATION:")
    te = rec["target_evaluation"]
    print(f"    Energy : {te['energy_status']} (gap: {te['energy_gap_pct']:+.1f}%)")
    print(f"    Carbon : {te['carbon_status']} (gap: {te['carbon_gap_pct']:+.1f}%)")


if __name__ == "__main__":
    for scenario in config.OPTIMIZATION_SCENARIOS.keys():
        rec = generate_recommendation(scenario)
        print_recommendation(rec)