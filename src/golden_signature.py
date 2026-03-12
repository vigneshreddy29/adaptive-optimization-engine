# =============================================================================
# src/golden_signature.py — Golden Signature Store & Update Logic
# =============================================================================

import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def initialize_golden_signatures(optimization_results):
    """
    Create the initial Golden Signature store from optimization results.
    Only initializes if file does not already exist (preserves updates).
    """
    # If GS file already exists, do not overwrite — preserve versions
    if os.path.exists(config.GOLDEN_SIGNATURE_FILE):
        return load_golden_signatures()

    print("[GS] Initializing Golden Signature store...")
    gs_store = {}

    for scenario_key, result in optimization_results.items():
        gs_store[scenario_key] = {
            "scenario_key"     : scenario_key,
            "scenario_label"   : result["scenario_label"],
            "version"          : 1,
            "created_at"       : datetime.now().isoformat(),
            "last_updated"     : datetime.now().isoformat(),
            "source_batch_id"  : result["best_batch_id"],
            "composite_score"  : result["composite_score"],
            "quality_score"    : result["quality_score"],
            "yield_score"      : result["yield_score"],
            "performance_score": result["performance_score"],
            "energy_kwh"       : result["energy_kwh"],
            "carbon_kg"        : result["carbon_kg"],
            "is_pareto"        : result["is_pareto"],
            "process_params"   : result["process_params"],
            "update_history"   : [],
            "hitl_overrides"   : [],
        }
        print(f"    GS created : {scenario_key} → Batch {result['best_batch_id']} "
              f"(score: {result['composite_score']})")

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    save_golden_signatures(gs_store)
    print(f"[GS] Store saved to {config.GOLDEN_SIGNATURE_FILE}")
    return gs_store


def save_golden_signatures(gs_store):
    """Persist Golden Signature store to disk as JSON."""
    with open(config.GOLDEN_SIGNATURE_FILE, "w") as f:
        json.dump(gs_store, f, indent=2)


def load_golden_signatures():
    """Load Golden Signature store from disk."""
    if not os.path.exists(config.GOLDEN_SIGNATURE_FILE):
        raise FileNotFoundError(
            f"Golden Signature file not found: {config.GOLDEN_SIGNATURE_FILE}\n"
            "Run the optimization engine first."
        )
    with open(config.GOLDEN_SIGNATURE_FILE, "r") as f:
        return json.load(f)


def normalize_batch_for_dominance(batch_metrics, reference_df):
    """
    Normalize a new batch's raw metrics to 0-1 scale
    using the reference dataset's min/max for fair comparison.

    Args:
        batch_metrics : dict with keys:
                        quality_score, yield_score, performance_score,
                        energy_kwh, carbon_kg
        reference_df  : the full processed DataFrame (60 batches)

    Returns:
        dict with norm_quality, norm_yield, norm_performance,
              norm_energy, norm_carbon
    """
    import pandas as pd

    def minmax_normalize(value, series, invert=False):
        min_v = series.min()
        max_v = series.max()
        if max_v == min_v:
            return 0.5
        # Clamp value to observed range before normalizing
        value = max(float(min_v), min(float(max_v), float(value)))
        result = (value - min_v) / (max_v - min_v)
        return (1 - result) if invert else result

    return {
        "norm_quality"    : minmax_normalize(
            batch_metrics["quality_score"],
            reference_df["Quality_Score"]
        ),
        "norm_yield"      : minmax_normalize(
            batch_metrics["yield_score"],
            reference_df["Yield_Score"]
        ),
        "norm_performance": minmax_normalize(
            batch_metrics["performance_score"],
            reference_df["Performance_Score"]
        ),
        "norm_energy"     : minmax_normalize(
            batch_metrics["energy_kwh"],
            reference_df["Est_Total_Energy_kWh"],
            invert=True   # lower energy = better
        ),
        "norm_carbon"     : minmax_normalize(
            batch_metrics["carbon_kg"],
            reference_df["Est_Carbon_kg"],
            invert=True   # lower carbon = better
        ),
    }


def compute_dominance_score(norm_metrics, weights):
    """
    Compute weighted composite score from normalized metrics.

    Args:
        norm_metrics : dict with norm_quality, norm_yield, etc.
        weights      : dict with w_quality, w_yield, etc.

    Returns:
        float composite score (0-1)
    """
    return (
        weights["w_quality"]      * norm_metrics["norm_quality"]     +
        weights["w_yield"]        * norm_metrics["norm_yield"]       +
        weights["w_performance"]  * norm_metrics["norm_performance"] +
        weights["w_energy"]       * norm_metrics["norm_energy"]      +
        weights["w_carbon"]       * norm_metrics["norm_carbon"]
    )


def evaluate_for_gs_update(scenario_key, new_batch_metrics,
                            reference_df, hitl_decision):
    """
    Full evaluation pipeline for GS update.

    This is called AFTER a HITL decision is made.
    It normalizes the new batch, computes its score,
    compares to current GS, and updates if warranted.

    Args:
        scenario_key      : which GS scenario to evaluate
        new_batch_metrics : dict with batch_id, quality_score, yield_score,
                            performance_score, energy_kwh, carbon_kg,
                            process_params
        reference_df      : full processed DataFrame for normalization
        hitl_decision     : "ACCEPT" / "REJECT" / "REPRIORITIZE"

    Returns:
        dict with:
            updated         : bool
            message         : str
            old_score       : float
            new_score       : float
            improvement_pct : float
            new_version     : int
    """
    if hitl_decision != "ACCEPT":
        return {
            "updated"        : False,
            "message"        : f"HITL decision was '{hitl_decision}' — GS update skipped.",
            "old_score"      : None,
            "new_score"      : None,
            "improvement_pct": 0.0,
            "new_version"    : None,
        }

    gs_store = load_golden_signatures()

    if scenario_key not in gs_store:
        return {
            "updated" : False,
            "message" : f"Scenario '{scenario_key}' not found in GS store.",
            "old_score": None, "new_score": None,
            "improvement_pct": 0.0, "new_version": None,
        }

    current_gs = gs_store[scenario_key]
    weights    = config.OPTIMIZATION_SCENARIOS[scenario_key]

    # Normalize new batch metrics using reference dataset
    norm_new = normalize_batch_for_dominance(new_batch_metrics, reference_df)

    # Compute new batch composite score
    new_score = round(compute_dominance_score(norm_new, weights), 4)
    old_score = current_gs["composite_score"]

    improvement = (new_score - old_score) / max(old_score, 1e-9)

    if improvement < config.GS_UPDATE_THRESHOLD:
        return {
            "updated"        : False,
            "message"        : (
                f"New batch score ({new_score:.4f}) does not exceed GS "
                f"({old_score:.4f}) by threshold "
                f"({config.GS_UPDATE_THRESHOLD*100:.1f}%). GS unchanged."
            ),
            "old_score"      : old_score,
            "new_score"      : new_score,
            "improvement_pct": round(improvement * 100, 2),
            "new_version"    : current_gs["version"],
        }

    # Archive current GS to history
    history_entry = {
        "archived_at"    : datetime.now().isoformat(),
        "version"        : current_gs["version"],
        "source_batch_id": current_gs["source_batch_id"],
        "composite_score": current_gs["composite_score"],
        "quality_score"  : current_gs["quality_score"],
        "yield_score"    : current_gs["yield_score"],
        "energy_kwh"     : current_gs["energy_kwh"],
        "carbon_kg"      : current_gs["carbon_kg"],
        "reason"         : "HITL ACCEPT — superseded by better batch",
    }
    current_gs["update_history"].append(history_entry)

    # Update GS to new best
    new_version = current_gs["version"] + 1
    current_gs.update({
        "version"          : new_version,
        "last_updated"     : datetime.now().isoformat(),
        "source_batch_id"  : new_batch_metrics.get("batch_id", "UNKNOWN"),
        "composite_score"  : new_score,
        "quality_score"    : new_batch_metrics.get("quality_score", 0),
        "yield_score"      : new_batch_metrics.get("yield_score", 0),
        "performance_score": new_batch_metrics.get("performance_score", 0),
        "energy_kwh"       : new_batch_metrics.get("energy_kwh", 0),
        "carbon_kg"        : new_batch_metrics.get("carbon_kg", 0),
        "process_params"   : new_batch_metrics.get("process_params", {}),
    })

    gs_store[scenario_key] = current_gs
    save_golden_signatures(gs_store)

    return {
        "updated"        : True,
        "message"        : (
            f"✅ GS UPDATED to v{new_version}! "
            f"Score: {old_score:.4f} → {new_score:.4f} "
            f"(+{improvement*100:.2f}%)"
        ),
        "old_score"      : old_score,
        "new_score"      : new_score,
        "improvement_pct": round(improvement * 100, 2),
        "new_version"    : new_version,
    }


def log_hitl_decision(batch_id, scenario_key, decision,
                      reason, weights_used):
    """
    Log every Human-in-the-Loop decision permanently.
    Reused for future decisions and audit trail.
    """
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)

    if os.path.exists(config.HITL_LOG_FILE):
        with open(config.HITL_LOG_FILE, "r") as f:
            log = json.load(f)
    else:
        log = []

    entry = {
        "timestamp"   : datetime.now().isoformat(),
        "batch_id"    : batch_id,
        "scenario_key": scenario_key,
        "decision"    : decision,
        "reason"      : reason,
        "weights_used": weights_used,
    }
    log.append(entry)

    with open(config.HITL_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

    return entry


def get_hitl_history():
    """Return full HITL decision history."""
    if not os.path.exists(config.HITL_LOG_FILE):
        return []
    with open(config.HITL_LOG_FILE, "r") as f:
        return json.load(f)


def get_gs_summary():
    """Return a clean summary of all current Golden Signatures."""
    gs_store = load_golden_signatures()
    summary  = []
    for key, gs in gs_store.items():
        summary.append({
            "Scenario"     : gs["scenario_label"],
            "Best Batch"   : gs["source_batch_id"],
            "Version"      : gs["version"],
            "Score"        : gs["composite_score"],
            "Quality"      : gs["quality_score"],
            "Yield"        : gs["yield_score"],
            "Energy (kWh)" : gs["energy_kwh"],
            "Carbon (kg)"  : gs["carbon_kg"],
            "Last Updated" : gs["last_updated"][:10],
        })
    return summary


def reset_golden_signatures():
    """Delete GS file to force re-initialization. Use for testing only."""
    if os.path.exists(config.GOLDEN_SIGNATURE_FILE):
        os.remove(config.GOLDEN_SIGNATURE_FILE)
        print("[GS] Golden Signature store reset.")