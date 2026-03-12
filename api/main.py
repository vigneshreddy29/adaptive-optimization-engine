# =============================================================================
# api/main.py — FastAPI Integration Layer
# =============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.optimization_engine import run_optimization, normalize_scores, identify_pareto_front
from src.golden_signature import (
    load_golden_signatures, update_golden_signature,
    log_hitl_decision, get_hitl_history, get_gs_summary
)
from src.adaptive_targets import compute_adaptive_targets, evaluate_batch_against_targets
from src.recommendation import generate_recommendation

# =============================================================================
# APP INIT
# =============================================================================
app = FastAPI(
    title="Adaptive Optimization Engine API",
    description=(
        "AI-Driven Adaptive Optimization Engine for Energy-Efficient "
        "& Sustainable Manufacturing — Team VORTEX, YUVAAN Hackathon"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================
class OptimizeRequest(BaseModel):
    scenario_key: str
    custom_weights: Optional[Dict[str, float]] = None

class HITLDecisionRequest(BaseModel):
    batch_id: str
    scenario_key: str
    decision: str          # "ACCEPT" / "REJECT" / "REPRIORITIZE"
    reason: Optional[str] = ""
    weights_used: Optional[Dict[str, float]] = None

class BatchEvalRequest(BaseModel):
    batch_id: str
    scenario_key: str
    process_params: Dict[str, float]

# =============================================================================
# HELPER
# =============================================================================
def load_df():
    return pd.read_csv(config.PROCESSED_DATA_FILE)

# =============================================================================
# ROUTES
# =============================================================================

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "system": "Adaptive Optimization Engine",
        "team": "VORTEX",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
def health_check():
    try:
        df = load_df()
        gs = load_golden_signatures()
        return {
            "status": "healthy",
            "batches_loaded": len(df),
            "golden_signatures": len(gs),
            "scenarios": list(gs.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize", tags=["Optimization"])
def optimize(request: OptimizeRequest):
    """
    Run multi-objective optimization for a given scenario.
    Returns best batch, composite score, and process parameters.
    """
    if request.scenario_key not in config.OPTIMIZATION_SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario_key. Valid options: "
                   f"{list(config.OPTIMIZATION_SCENARIOS.keys())}"
        )
    try:
        results, _ = run_optimization()
        result = results.get(request.scenario_key)
        if not result:
            raise HTTPException(status_code=404, detail="Scenario not found.")
        return {
            "status": "success",
            "scenario_key": request.scenario_key,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/optimize/all", tags=["Optimization"])
def optimize_all():
    """Run optimization across all 3 scenarios and return all results."""
    try:
        results, full_df = run_optimization()
        full_df = normalize_scores(full_df)
        full_df, pareto_df = identify_pareto_front(full_df)
        return {
            "status": "success",
            "total_batches": len(full_df),
            "pareto_count": int(full_df["Is_Pareto"].sum()),
            "scenarios": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/golden-signature/{scenario_key}", tags=["Golden Signatures"])
def get_golden_signature(scenario_key: str):
    """Retrieve the current Golden Signature for a specific scenario."""
    try:
        gs_store = load_golden_signatures()
        if scenario_key not in gs_store:
            raise HTTPException(
                status_code=404,
                detail=f"Scenario '{scenario_key}' not found. "
                       f"Valid: {list(gs_store.keys())}"
            )
        return {
            "status": "success",
            "golden_signature": gs_store[scenario_key]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/golden-signature", tags=["Golden Signatures"])
def get_all_golden_signatures():
    """Retrieve all Golden Signatures with summary."""
    try:
        summary = get_gs_summary()
        return {
            "status": "success",
            "count": len(summary),
            "golden_signatures": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hitl-decision", tags=["HITL"])
def submit_hitl_decision(request: HITLDecisionRequest):
    """
    Submit a Human-in-the-Loop decision.
    Decisions: ACCEPT / REJECT / REPRIORITIZE
    All decisions are logged and reused for future recommendations.
    """
    valid_decisions = ["ACCEPT", "REJECT", "REPRIORITIZE"]
    if request.decision not in valid_decisions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decision. Must be one of: {valid_decisions}"
        )
    try:
        weights = (request.weights_used or
                   config.OPTIMIZATION_SCENARIOS.get(request.scenario_key, {}))
        entry = log_hitl_decision(
            batch_id=request.batch_id,
            scenario_key=request.scenario_key,
            decision=request.decision,
            reason=request.reason or "",
            weights_used=weights
        )
        return {
            "status": "success",
            "message": f"Decision '{request.decision}' logged for batch "
                       f"{request.batch_id}",
            "entry": entry
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hitl-history", tags=["HITL"])
def get_hitl_decision_history():
    """Retrieve full HITL decision history for audit and learning."""
    try:
        history = get_hitl_history()
        return {
            "status": "success",
            "total_decisions": len(history),
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", tags=["Recommendations"])
def get_recommendation(request: BatchEvalRequest):
    """
    Generate actionable parameter recommendations by comparing
    a batch configuration against the Golden Signature.
    """
    if request.scenario_key not in config.OPTIMIZATION_SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario_key."
        )
    try:
        rec = generate_recommendation(
            request.scenario_key,
            request.process_params
        )
        return {"status": "success", "recommendation": rec}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/adaptive-targets", tags=["Targets"])
def get_adaptive_targets(regulatory_pressure: float = 1.0):
    """
    Get current adaptive energy and carbon targets.
    regulatory_pressure: 1.0=normal, 1.5=high, 2.0=critical
    """
    try:
        df = load_df()
        targets = compute_adaptive_targets(df, regulatory_pressure)
        return {"status": "success", "targets": targets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", tags=["History"])
def get_batch_history():
    """Return all 60 processed batches with scores."""
    try:
        df = load_df()
        display_cols = [
            "Batch_ID", "Quality_Score", "Yield_Score",
            "Performance_Score", "Est_Total_Energy_kWh", "Est_Carbon_kg"
        ]
        records = df[display_cols].to_dict(orient="records")
        return {
            "status": "success",
            "total_batches": len(records),
            "batches": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))