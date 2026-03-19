# =============================================================================
# dashboard/app.py — Streamlit HITL Dashboard
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_pipeline import run_pipeline, load_timeseries_data
from src.optimization_engine import run_optimization, normalize_scores, identify_pareto_front
from src.golden_signature import (
    initialize_golden_signatures, load_golden_signatures,
    evaluate_for_gs_update, log_hitl_decision,
    get_hitl_history, get_gs_summary
)
from src.adaptive_targets import compute_adaptive_targets, evaluate_batch_against_targets
from src.recommendation import generate_recommendation
from src.feature_engineering import (
    models_exist, load_models, train_quality_model,
    train_energy_model, save_models, predict_outcomes,
    build_engineered_features
)

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Adaptive Optimization Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# STYLING
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #1e3a5f, #2196F3);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem; color: #666; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa; border-left: 4px solid #2196F3;
        padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;
    }
    .gs-card {
        background: #e8f4fd; border: 1px solid #2196F3;
        padding: 1rem; border-radius: 8px; margin-bottom: 0.8rem;
    }
    .hitl-accept  { background: #e8f5e9; border-left: 4px solid #4CAF50; padding: 0.8rem; border-radius: 6px; }
    .hitl-reject  { background: #ffebee; border-left: 4px solid #f44336; padding: 0.8rem; border-radius: 6px; }
    .hitl-reprio  { background: #fff3e0; border-left: 4px solid #FF9800; padding: 0.8rem; border-radius: 6px; }
    .on-target    { color: #4CAF50; font-weight: bold; }
    .above-target { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "df" not in st.session_state:
        st.session_state.df = None
    if "optimization_results" not in st.session_state:
        st.session_state.optimization_results = None
    if "gs_store" not in st.session_state:
        st.session_state.gs_store = None
    if "hitl_log" not in st.session_state:
        st.session_state.hitl_log = []
    if "custom_weights" not in st.session_state:
        st.session_state.custom_weights = None

init_session_state()

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_all_data():
    df = run_pipeline()
    results, full_df = run_optimization()
    full_df = normalize_scores(full_df)
    full_df, _ = identify_pareto_front(full_df)
    gs_store = initialize_golden_signatures(results)
    return df, results, full_df, gs_store

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## ⚡ Optimization Engine")
    st.markdown("**Team VORTEX** | YUVAAN Hackathon")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "📊 Data Explorer",
         "🎯 Optimization Engine",
         "🏆 Golden Signatures",
         "🤝 HITL Workflow",
         "🧠 Model Intelligence",
         "🔬 T001 Analysis",
         "📈 History & Learning"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if st.button("🔄 Load / Refresh Data", use_container_width=True):
        st.cache_data.clear()
        with st.spinner("Running pipeline..."):
            (st.session_state.df,
             st.session_state.optimization_results,
             st.session_state.full_df,
             st.session_state.gs_store) = load_all_data()
            st.session_state.data_loaded = True
        st.success("Data loaded successfully.")

    if st.session_state.data_loaded:
        st.markdown("✅ **Data Status:** Loaded")
        st.markdown(f"📦 **Batches:** {len(st.session_state.df)}")
        st.markdown(f"🏆 **GS Scenarios:** 3")
    else:
        st.markdown("⚠️ **Click Load Data to begin**")

    st.markdown("---")
    st.markdown("**⚙️ Regulatory Pressure**")
    st.session_state.reg_pressure = st.slider(
        "Regulatory Pressure",
        min_value=0.0, max_value=1.0,
        value=st.session_state.get("reg_pressure", 0.5),
        step=0.05,
        help="0 = relaxed targets · 1 = maximum regulatory stringency"
    )
    pressure_label = (
        "🟢 Relaxed" if st.session_state.reg_pressure < 0.3 else
        "🟡 Moderate" if st.session_state.reg_pressure < 0.7 else
        "🔴 Stringent"
    )
    st.caption(f"Mode: {pressure_label}")

# =============================================================================
# AUTO LOAD
# =============================================================================
if not st.session_state.data_loaded:
    with st.spinner("Initializing system..."):
        (st.session_state.df,
         st.session_state.optimization_results,
         st.session_state.full_df,
         st.session_state.gs_store) = load_all_data()
        st.session_state.data_loaded = True

df       = st.session_state.df
full_df  = st.session_state.full_df
results  = st.session_state.optimization_results
gs_store = st.session_state.gs_store

# =============================================================================
# PAGE 1 — OVERVIEW
# =============================================================================
if page == "🏠 Overview":
    st.markdown('<div class="main-header">AI-Driven Adaptive Optimization Engine</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Energy-Efficient & Sustainable Manufacturing | Team VORTEX</div>',
                unsafe_allow_html=True)

    targets = compute_adaptive_targets(df, regulatory_pressure=st.session_state.get("reg_pressure", 0.5))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Batches",      f"{len(df)}")
    col2.metric("Pareto Optimal",     f"{full_df['Is_Pareto'].sum()}")
    col3.metric("Avg Energy (kWh)",   f"{df['Est_Total_Energy_kWh'].mean():.1f}")
    col4.metric("Avg Carbon (kg)",    f"{df['Est_Carbon_kg'].mean():.2f}")
    col5.metric("Best Energy (kWh)",  f"{df['Est_Total_Energy_kWh'].min():.1f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📉 Energy Distribution Across Batches")
        fig = px.histogram(df, x="Est_Total_Energy_kWh",
                           nbins=15, color_discrete_sequence=["#2196F3"],
                           labels={"Est_Total_Energy_kWh": "Energy (kWh)"})
        fig.add_vline(x=targets["target_energy_kwh"], line_dash="dash",
                      line_color="red", annotation_text="Target")
        fig.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("🌿 Quality vs Energy Trade-off")
        fig2 = px.scatter(full_df,
                          x="Est_Total_Energy_kWh",
                          y="Quality_Score",
                          color="Is_Pareto",
                          hover_data=["Batch_ID"],
                          color_discrete_map={True: "#4CAF50", False: "#90CAF9"},
                          labels={"Est_Total_Energy_kWh": "Energy (kWh)",
                                  "Quality_Score": "Quality Score",
                                  "Is_Pareto": "Pareto Optimal"})
        fig2.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 Golden Signature Summary")
    gs_summary = get_gs_summary()
    st.dataframe(pd.DataFrame(gs_summary), use_container_width=True)

# =============================================================================
# PAGE 2 — DATA EXPLORER
# =============================================================================
elif page == "📊 Data Explorer":
    st.header("📊 Batch Data Explorer")

    tab1, tab2, tab3 = st.tabs(["Batch Scores", "Process Parameters", "Correlations"])

    with tab1:
        st.subheader("All 60 Batches — Performance Scores")
        display_cols = ["Batch_ID", "Quality_Score", "Yield_Score",
                        "Performance_Score", "Est_Total_Energy_kWh",
                        "Est_Carbon_kg", "Is_Pareto"]
        st.dataframe(
            full_df[display_cols].sort_values("Quality_Score", ascending=False),
            use_container_width=True
        )

    with tab2:
        st.subheader("Process Parameters Distribution")
        param = st.selectbox("Select Parameter", config.PROCESS_PARAM_COLS)
        fig = px.box(df, y=param, points="all",
                     color_discrete_sequence=["#2196F3"])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Correlation Heatmap")
        corr_cols = config.PROCESS_PARAM_COLS + [
            "Quality_Score", "Yield_Score",
            "Est_Total_Energy_kWh", "Est_Carbon_kg"
        ]
        corr = df[corr_cols].corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE 3 — OPTIMIZATION ENGINE
# =============================================================================
elif page == "🎯 Optimization Engine":
    st.header("🎯 Multi-Objective Optimization Engine")

    st.subheader("Pareto Front Visualization")
    fig = px.scatter_3d(
        full_df,
        x="Quality_Score",
        y="Yield_Score",
        z="Est_Total_Energy_kWh",
        color="Is_Pareto",
        hover_data=["Batch_ID", "Est_Carbon_kg"],
        color_discrete_map={True: "#4CAF50", False: "#90CAF9"},
        labels={"Is_Pareto": "Pareto Optimal"}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Scenario Results")

    for scenario_key, result in results.items():
        with st.expander(f"📌 {result['scenario_label']}", expanded=True):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Best Batch",   result["best_batch_id"])
            c2.metric("Score",        result["composite_score"])
            c3.metric("Quality",      f"{result['quality_score']:.1f}")
            c4.metric("Energy (kWh)", f"{result['energy_kwh']:.1f}")
            c5.metric("Carbon (kg)",  f"{result['carbon_kg']:.3f}")

# =============================================================================
# PAGE 4 — GOLDEN SIGNATURES
# =============================================================================
elif page == "🏆 Golden Signatures":
    st.header("🏆 Golden Signature Framework")
    st.markdown("Each Golden Signature represents the best-known batch "
                "configuration for a specific optimization objective.")

    gs_store_current = load_golden_signatures()

    for key, gs in gs_store_current.items():
        with st.expander(f"🥇 {gs['scenario_label']} — v{gs['version']}",
                         expanded=True):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"**Source Batch:** `{gs['source_batch_id']}`")
                st.markdown(f"**Version:** {gs['version']}")
                st.markdown(f"**Last Updated:** {gs['last_updated'][:10]}")
                st.markdown(f"**Composite Score:** {gs['composite_score']}")

                metrics_data = {
                    "Metric": ["Quality", "Yield", "Performance",
                               "Energy (kWh)", "Carbon (kg)"],
                    "Value": [gs["quality_score"], gs["yield_score"],
                              gs["performance_score"],
                              gs["energy_kwh"], gs["carbon_kg"]]
                }
                st.dataframe(pd.DataFrame(metrics_data),
                             use_container_width=True, hide_index=True)

            with col2:
                st.markdown("**Process Parameters:**")
                params_df = pd.DataFrame([
                    {"Parameter": k, "Value": v}
                    for k, v in gs["process_params"].items()
                ])
                st.dataframe(params_df, use_container_width=True,
                             hide_index=True)

            if gs["update_history"]:
                st.markdown(f"**Update History:** {len(gs['update_history'])} "
                            f"previous versions archived")

# =============================================================================
# PAGE 5 — HITL WORKFLOW
# =============================================================================
elif page == "🤝 HITL Workflow":
    st.header("🤝 Human-in-the-Loop Workflow")
    st.markdown("Configure a new batch, compare against the Golden Signature, "
                "and make a binding decision that updates the system.")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("⚙️ Step 1 — Configure New Batch")

        scenario_key = st.selectbox(
            "Optimization Scenario",
            list(config.OPTIMIZATION_SCENARIOS.keys()),
            format_func=lambda x: config.OPTIMIZATION_SCENARIOS[x]["label"]
        )

        st.markdown("**Adjust Process Parameters:**")
        new_params = {}
        gs_store_c = load_golden_signatures()
        gs_ref     = gs_store_c[scenario_key]["process_params"]

        for param in config.PROCESS_PARAM_COLS:
            col_range = df[param]
            new_params[param] = st.slider(
                param,
                min_value=float(col_range.min()),
                max_value=float(col_range.max()),
                value=float(col_range.mean()),
                step=float((col_range.max() - col_range.min()) / 100),
                key=f"slider_{param}"
            )

        batch_id_input = st.text_input("New Batch ID", value="T_NEW_001")

        # ── Z-score signals for each parameter ──────────────────────────────
        gran_z   = (new_params["Granulation_Time"]  - df["Granulation_Time"].mean())  / df["Granulation_Time"].std()
        binder_z = (new_params["Binder_Amount"]      - df["Binder_Amount"].mean())     / df["Binder_Amount"].std()
        dtime_z  = (new_params["Drying_Time"]        - df["Drying_Time"].mean())       / df["Drying_Time"].std()
        dtemp_z  = (new_params["Drying_Temp"]        - df["Drying_Temp"].mean())       / df["Drying_Temp"].std()
        comp_z   = (new_params["Compression_Force"]  - df["Compression_Force"].mean()) / df["Compression_Force"].std()
        speed_z  = (new_params["Machine_Speed"]      - df["Machine_Speed"].mean())     / df["Machine_Speed"].std()
        moist_z  = (new_params["Moisture_Content"]   - df["Moisture_Content"].mean())  / df["Moisture_Content"].std()
        lub_z    = (new_params["Lubricant_Conc"]     - df["Lubricant_Conc"].mean())    / df["Lubricant_Conc"].std()

        # ── Quality signal (correlation-weighted) ───────────────────────────
        quality_signal = (
            + 0.984 * gran_z
            + 0.984 * binder_z
            + 0.984 * dtime_z
            + 0.983 * speed_z
            + 0.970 * lub_z
            - 0.984 * dtemp_z
            - 0.985 * comp_z
            - 0.988 * moist_z
        ) / 8.0

        sim_quality = float(df["Quality_Score"].mean()) + quality_signal * df["Quality_Score"].std() * 2.5
        sim_quality = max(40.0, min(75.0, sim_quality))

        # ── Yield proxy — driven by same quality signal ──────────────────────
        # High quality parameters → high yield (consistent tablets)
        sim_yield = float(df["Yield_Score"].mean()) + (quality_signal * df["Yield_Score"].std() * 3.0)
        sim_yield = max(5.0, min(99.9, sim_yield))

        # ── Performance proxy — moisture deviation from optimal 2.0% ─────────
        optimal_moist   = 2.0
        moist_dev       = abs(new_params["Moisture_Content"] - optimal_moist)
        max_mdev        = float(df["Moisture_Content"].apply(
                              lambda x: abs(x - optimal_moist)).max())
        sim_performance = (1 - moist_dev / max(max_mdev, 1e-9)) * 100
        sim_performance = max(5.0, min(100.0, sim_performance))

        # ── Energy estimation ────────────────────────────────────────────────
        gran_e  = (new_params["Granulation_Time"] / config.T001_GRANULATION_TIME) * \
                  config.PHASE_ENERGY_PROPORTIONS["Granulation"] * config.T001_TOTAL_ENERGY
        dry_e   = (new_params["Drying_Time"]  / config.T001_DRYING_TIME) * \
                  (new_params["Drying_Temp"]  / config.T001_DRYING_TEMP) * \
                  config.PHASE_ENERGY_PROPORTIONS["Drying"] * config.T001_TOTAL_ENERGY
        comp_e  = (new_params["Compression_Force"] / config.T001_COMPRESSION_FORCE) * \
                  (new_params["Machine_Speed"]      / config.T001_MACHINE_SPEED) * \
                  config.PHASE_ENERGY_PROPORTIONS["Compression"] * config.T001_TOTAL_ENERGY
        other_e = (new_params["Machine_Speed"] / config.T001_MACHINE_SPEED) * \
                  (config.PHASE_ENERGY_PROPORTIONS["Milling"] +
                   config.PHASE_ENERGY_PROPORTIONS["Coating"] +
                   config.PHASE_ENERGY_PROPORTIONS["Blending"] +
                   config.PHASE_ENERGY_PROPORTIONS["Quality_Testing"] +
                   config.PHASE_ENERGY_PROPORTIONS["Preparation"]) * config.T001_TOTAL_ENERGY

        sim_energy_kwh = (gran_e + dry_e + comp_e + other_e) / 60.0
        sim_carbon_kg  = sim_energy_kwh * config.CARBON_EMISSION_FACTOR * 1000

        st.markdown("**Simulated Outcomes for This Configuration:**")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Est. Quality",    f"{sim_quality:.1f}")
        sc2.metric("Est. Energy kWh", f"{sim_energy_kwh:.1f}")
        sc3.metric("Est. Carbon kg",  f"{sim_carbon_kg:.3f}")

    with col_right:
        st.subheader("📊 Step 2 — Compare vs Golden Signature")

        rec = generate_recommendation(scenario_key, new_params)

        gs_current = load_golden_signatures()[scenario_key]
        st.markdown(
            f"**Golden Batch:** `{rec['golden_batch_id']}` "
            f"(v{gs_current['version']}, score: {rec['gs_composite_score']})"
        )

        st.markdown("**Parameter Gap Analysis:**")
        adj_data = []
        for param, adj in rec["parameter_adjustments"].items():
            arrow = {"INCREASE": "↑", "DECREASE": "↓",
                     "MAINTAIN": "→"}[adj["direction"]]
            adj_data.append({
                "Parameter"    : param,
                "Your Value"   : adj["current"],
                "GS Recommend" : adj["recommended"],
                "Change"       : f"{arrow} {adj['delta_pct']:+.1f}%"
            })
        st.dataframe(pd.DataFrame(adj_data),
                     use_container_width=True, hide_index=True)

        ei = rec["expected_impact"]
        st.markdown("**Expected Impact if GS Parameters Adopted:**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Energy Saving",  f"{ei['energy_saving_pct']:+.1f}%")
        m2.metric("Carbon Saving",  f"{ei['carbon_saving_pct']:+.1f}%")
        m3.metric("Quality Δ",      f"{ei['quality_improvement']:+.1f}")
        m4.metric("Yield Δ",        f"{ei['yield_improvement']:+.1f}")

        te = rec["target_evaluation"]
        st.markdown("**Target Evaluation:**")
        st.markdown(
            f"Energy: {'🟢' if '✅' in te['energy_status'] else '🔴'} "
            f"{te['energy_status']} ({te['energy_gap_pct']:+.1f}%)  |  "
            f"Carbon: {'🟢' if '✅' in te['carbon_status'] else '🔴'} "
            f"{te['carbon_status']} ({te['carbon_gap_pct']:+.1f}%)"
        )

    st.markdown("---")
    st.subheader("✋ Step 3 — Human Decision")

    reason = st.text_input(
        "Reason for decision (optional)",
        placeholder="e.g. Parameters within safe range, operator approved"
    )

    col_d1, col_d2, col_d3 = st.columns(3)

    # ── Build new batch metrics for GS evaluation ────────────────────────────
    new_batch_metrics = {
        "batch_id"         : batch_id_input,
        "quality_score"    : round(sim_quality, 2),
        "yield_score"      : round(sim_yield, 2),
        "performance_score": round(sim_performance, 2),
        "energy_kwh"       : round(sim_energy_kwh, 2),
        "carbon_kg"        : round(sim_carbon_kg, 4),
        "process_params"   : new_params,
    }

    with col_d1:
        if st.button("✅ ACCEPT — Use GS Parameters",
                     use_container_width=True, type="primary"):

            log_hitl_decision(
                batch_id=batch_id_input,
                scenario_key=scenario_key,
                decision="ACCEPT",
                reason=reason or "Operator accepted GS recommendation",
                weights_used=config.OPTIMIZATION_SCENARIOS[scenario_key]
            )

            from src.golden_signature import evaluate_for_gs_update
            update_result = evaluate_for_gs_update(
                scenario_key=scenario_key,
                new_batch_metrics=new_batch_metrics,
                reference_df=df,
                hitl_decision="ACCEPT"
            )

            if update_result["updated"]:
                st.success(update_result["message"])
                st.balloons()
                st.info(
                    f"📈 GS Version: {update_result['old_score']:.4f} → "
                    f"{update_result['new_score']:.4f} "
                    f"(+{update_result['improvement_pct']:.2f}%)"
                )
                st.session_state.gs_store = load_golden_signatures()
            else:
                st.warning(update_result["message"])
                st.info(
                    f"Current GS score: {update_result['old_score']:.4f} | "
                    f"Your batch score: {update_result['new_score']:.4f}"
                    if update_result["new_score"] else
                    update_result["message"]
                )

            st.session_state.hitl_log = get_hitl_history()

    with col_d2:
        if st.button("❌ REJECT — Keep Current Parameters",
                     use_container_width=True):
            log_hitl_decision(
                batch_id=batch_id_input,
                scenario_key=scenario_key,
                decision="REJECT",
                reason=reason or "Operator rejected GS recommendation",
                weights_used=config.OPTIMIZATION_SCENARIOS[scenario_key]
            )
            st.warning(f"❌ Decision logged: REJECT for `{batch_id_input}`")
            st.info("Golden Signature unchanged.")
            st.session_state.hitl_log = get_hitl_history()

    with col_d3:
        if st.button("🔄 REPRIORITIZE — Adjust Weights",
                     use_container_width=True):
            st.session_state.show_reprio = True

    if st.session_state.get("show_reprio", False):
        st.markdown("---")
        st.markdown("**Reprioritize Objectives — Set New Weights:**")
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        w_q = rc1.slider("Quality",     0.0, 1.0, 0.20, 0.05, key="w_q")
        w_y = rc2.slider("Yield",       0.0, 1.0, 0.20, 0.05, key="w_y")
        w_p = rc3.slider("Performance", 0.0, 1.0, 0.20, 0.05, key="w_p")
        w_e = rc4.slider("Energy",      0.0, 1.0, 0.20, 0.05, key="w_e")
        w_c = rc5.slider("Carbon",      0.0, 1.0, 0.20, 0.05, key="w_c")

        total_w = w_q + w_y + w_p + w_e + w_c
        st.markdown(
            f"**Weight Total:** {total_w:.2f} "
            f"{'✅ Valid' if abs(total_w - 1.0) < 0.05 else '⚠️ Should sum to ~1.0'}"
        )

        if st.button("Apply Reprioritization", type="primary"):
            custom_weights = {
                "w_quality": w_q, "w_yield": w_y,
                "w_performance": w_p, "w_energy": w_e, "w_carbon": w_c
            }
            log_hitl_decision(
                batch_id=batch_id_input,
                scenario_key=scenario_key,
                decision="REPRIORITIZE",
                reason=reason or
                       f"Custom weights: Q={w_q} Y={w_y} P={w_p} E={w_e} C={w_c}",
                weights_used=custom_weights
            )

            from src.golden_signature import evaluate_for_gs_update
            update_result = evaluate_for_gs_update(
                scenario_key=scenario_key,
                new_batch_metrics=new_batch_metrics,
                reference_df=df,
                hitl_decision="ACCEPT"
            )

            st.success("✅ Reprioritization logged and applied.")
            if update_result["updated"]:
                st.success(update_result["message"])
                st.session_state.gs_store = load_golden_signatures()
            st.session_state.show_reprio = False
            st.session_state.hitl_log = get_hitl_history()
# =============================================================================
# PAGE 6 — MODEL INTELLIGENCE
# =============================================================================
elif page == "🧠 Model Intelligence":
    st.header("🧠 Model Intelligence — XGBoost Predictive Engine")
    st.markdown("Two XGBoost models trained on 60 batches predict quality "
                "and energy outcomes from process parameters.")

    st.markdown("---")

    # Load or train models
    if not models_exist():
        with st.spinner("Training models..."):
            df_fe = build_engineered_features(df)
            q_model, q_scaler, q_metrics = train_quality_model(df)
            e_model, e_scaler, e_metrics = train_energy_model(df)
            save_models(q_model, q_scaler, q_metrics,
                        e_model, e_scaler, e_metrics)
        st.success("Models trained and saved.")

    _, _, _, _, metrics = load_models()
    qm = metrics["quality_model"]
    em = metrics["energy_model"]

    # --- Model Performance Cards ---
    st.subheader("📊 Model Performance")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Quality Prediction Model")
        st.markdown("**Target:** Dissolution Rate → Quality Score")
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Train",   f"{qm['r2_train']:.4f}")
        m2.metric("R² CV",      f"{qm['r2_cv_mean']:.4f} ± {qm['r2_cv_std']:.4f}")
        m3.metric("RMSE",       f"{qm['rmse_train']:.4f}")

        st.markdown("**Top Feature Importances:**")
        fi_q = qm["feature_importance"]
        fi_q_df = pd.DataFrame([
            {"Feature": k, "Importance": v}
            for k, v in fi_q.items()
        ])
        import plotly.express as px
        fig_q = px.bar(fi_q_df, x="Importance", y="Feature",
                       orientation="h",
                       color="Importance",
                       color_continuous_scale="Blues")
        fig_q.update_layout(height=300, margin=dict(t=10, b=10),
                            showlegend=False)
        st.plotly_chart(fig_q, use_container_width=True)

    with col2:
        st.markdown("### ⚡ Energy Prediction Model")
        st.markdown("**Target:** Total Energy (kWh)")
        m4, m5, m6 = st.columns(3)
        m4.metric("R² Train",   f"{em['r2_train']:.4f}")
        m5.metric("R² CV",      f"{em['r2_cv_mean']:.4f} ± {em['r2_cv_std']:.4f}")
        m6.metric("RMSE",       f"{em['rmse_train']:.4f}")

        st.markdown("**Top Feature Importances:**")
        fi_e = em["feature_importance"]
        fi_e_df = pd.DataFrame([
            {"Feature": k, "Importance": v}
            for k, v in fi_e.items()
        ])
        fig_e = px.bar(fi_e_df, x="Importance", y="Feature",
                       orientation="h",
                       color="Importance",
                       color_continuous_scale="Oranges")
        fig_e.update_layout(height=300, margin=dict(t=10, b=10),
                            showlegend=False)
        st.plotly_chart(fig_e, use_container_width=True)

    st.markdown("---")

    # --- Live Prediction ---
    st.subheader("🔮 Live Prediction — Enter Process Parameters")
    st.markdown("Adjust parameters below to get instant AI-predicted outcomes.")

    pred_cols = st.columns(4)
    pred_params = {}
    param_list = config.PROCESS_PARAM_COLS
    for i, param in enumerate(param_list):
        with pred_cols[i % 4]:
            pred_params[param] = st.number_input(
                param,
                min_value=float(df[param].min()),
                max_value=float(df[param].max()),
                value=float(df[param].mean()),
                step=float((df[param].max() - df[param].min()) / 100),
                key=f"pred_{param}"
            )

    prediction = predict_outcomes(pred_params, df)

    if prediction:
        st.markdown("### 🎯 Predicted Outcomes")
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Predicted Quality Score",   f"{prediction['predicted_quality']:.1f}")
        pc2.metric("Predicted Dissolution Rate",f"{prediction['predicted_dissolution']:.1f}%")
        pc3.metric("Predicted Energy (kWh)",    f"{prediction['predicted_energy']:.1f}")
        pc4.metric("Predicted Carbon (kg)",     f"{prediction['predicted_carbon']:.3f}")

        # Compare vs fleet average
        st.markdown("### 📊 vs Fleet Average")
        avg_quality = df["Quality_Score"].mean()
        avg_energy  = df["Est_Total_Energy_kWh"].mean()
        avg_carbon  = df["Est_Carbon_kg"].mean()

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Quality vs Avg",
                   f"{prediction['predicted_quality']:.1f}",
                   f"{prediction['predicted_quality'] - avg_quality:+.1f}")
        cc2.metric("Energy vs Avg",
                   f"{prediction['predicted_energy']:.1f} kWh",
                   f"{prediction['predicted_energy'] - avg_energy:+.1f} kWh",
                   delta_color="inverse")
        cc3.metric("Carbon vs Avg",
                   f"{prediction['predicted_carbon']:.3f} kg",
                   f"{prediction['predicted_carbon'] - avg_carbon:+.3f} kg",
                   delta_color="inverse")

    st.markdown("---")

    # --- Actual vs Predicted Chart ---
    st.subheader("📈 Model Validation — Actual vs Predicted")

    q_model, q_scaler, e_model, e_scaler, _ = load_models()

    QUALITY_FEATURES = [
        "Granulation_Time", "Binder_Amount", "Drying_Time",
        "Machine_Speed", "Moisture_Content",
        "Compression_Force", "Lubricant_Conc",
    ]
    X_q      = q_scaler.transform(df[QUALITY_FEATURES].values)
    y_q_pred = q_model.predict(X_q)

    # Convert dissolution predictions back to quality score
    diss_mean = qm.get("dissolution_mean", 90.93)
    diss_std  = qm.get("dissolution_std",  4.63)
    q_mean    = qm.get("quality_mean",     54.94)
    q_std     = qm.get("quality_std",      2.85)
    y_q_score = q_mean + ((y_q_pred - diss_mean) / diss_std) * q_std

    val_df = pd.DataFrame({
        "Batch_ID"          : df["Batch_ID"],
        "Actual Quality"    : df["Quality_Score"].round(2),
        "Predicted Quality" : y_q_score.round(2),
    })

    fig_val = px.scatter(
        val_df,
        x="Actual Quality",
        y="Predicted Quality",
        hover_data=["Batch_ID"],
        color_discrete_sequence=["#2196F3"]
    )

    # Perfect prediction line (red dashed)
    q_min = float(val_df["Actual Quality"].min())
    q_max = float(val_df["Actual Quality"].max())
    fig_val.add_shape(
        type="line",
        x0=q_min, y0=q_min,
        x1=q_max, y1=q_max,
        line=dict(color="red", dash="dash", width=2),
    )

    # Trend line using numpy polyfit (orange)
    z       = np.polyfit(val_df["Actual Quality"], val_df["Predicted Quality"], 1)
    p       = np.poly1d(z)
    x_line  = np.linspace(q_min, q_max, 50)
    fig_val.add_scatter(
        x=x_line, y=p(x_line),
        mode="lines",
        line=dict(color="orange", width=2, dash="dot"),
        name="Trend"
    )

    fig_val.update_layout(height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig_val, use_container_width=True)
    st.caption(
        "🔵 Blue dots = model predictions | "
        "🔴 Red dashed = perfect prediction line | "
        "🟠 Orange dotted = actual trend"
    )
# =============================================================================
# PAGE 7 — T001 TIME-SERIES ANALYSIS
# =============================================================================
elif page == "🔬 T001 Analysis":
    st.header("🔬 T001 Batch — Time-Series Analysis")
    st.markdown("Deep dive into Batch T001's 211-point process time-series. "
                "Phase energy profiles, vibration anomaly detection, and efficiency ranking.")

    st.markdown("---")

    with st.spinner("Loading T001 time-series..."):
        ts_data = load_timeseries_data()

    if ts_data is None:
        st.error("Time-series file not found. Ensure `_h_batch_process_data.xlsx` is in `data/raw/`.")
        st.stop()

    ts_df       = ts_data["timeseries"]
    phase_sum   = ts_data["phase_summary"]
    n_anomalies = ts_data["n_anomalies"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Data Points",  f"{ts_data['n_points']}")
    k2.metric("Production Phases",  f"{ts_df['Phase_Label'].nunique()}")
    k3.metric("Anomalies Detected", f"{n_anomalies}",
              delta=f"{'⚠️ Review needed' if n_anomalies > 5 else '✅ Normal range'}",
              delta_color="off")
    k4.metric("Peak Power",
              f"{ts_df['Energy_kW'].max():.1f} kW",
              f"Phase: {ts_df.loc[ts_df['Energy_kW'].idxmax(), 'Phase_Label']}")

    st.markdown("---")
    st.subheader("📈 Power Consumption Over Time — All Phases")

    phase_colors = {
        "Compression"    : "#FF6B6B",
        "Drying"         : "#FFB800",
        "Milling"        : "#4ECDC4",
        "Coating"        : "#45B7D1",
        "Blending"       : "#96CEB4",
        "Granulation"    : "#00D4AA",
        "Quality_Testing": "#B96AFF",
        "Preparation"    : "#74B9FF",
    }

    fig_ts = go.Figure()
    for phase in ts_df["Phase_Label"].unique():
        mask     = ts_df["Phase_Label"] == phase
        phase_df = ts_df[mask]
        color    = phase_colors.get(phase, "#AAAAAA")
# Convert hex to rgba for fill transparency
        hex_color = color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        fill_color = f"rgba({r},{g},{b},0.15)"

        fig_ts.add_trace(go.Scatter(
            x=phase_df["Time_min"],
            y=phase_df["Energy_kW"],
            mode="lines",
            name=phase,
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=fill_color,
            hovertemplate=f"<b>{phase}</b><br>Time: %{{x:.1f}} min<br>Power: %{{y:.2f}} kW<extra></extra>"
        ))

    anomalies = ts_df[ts_df["Is_Anomaly"]]
    if not anomalies.empty:
        fig_ts.add_trace(go.Scatter(
            x=anomalies["Time_min"], y=anomalies["Energy_kW"],
            mode="markers", name="⚠️ Anomaly",
            marker=dict(color="red", size=10, symbol="x",
                        line=dict(color="darkred", width=2)),
            hovertemplate="<b>⚠️ ANOMALY</b><br>Time: %{x:.1f} min<br>Power: %{y:.2f} kW<extra></extra>"
        ))

    fig_ts.update_layout(
        height=420, xaxis_title="Time (minutes)", yaxis_title="Power Consumption (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=60, b=40, l=60, r=20), hovermode="x unified"
    )
    fig_ts.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
    fig_ts.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
    st.plotly_chart(fig_ts, use_container_width=True)

    if n_anomalies > 0:
        st.warning(f"⚠️ {n_anomalies} anomalous energy readings detected (Z-score > 2.0). "
                   f"See red × markers above.")

    st.markdown("---")
    st.subheader("📳 Vibration Profile — Phase Comparison")

    col_v1, col_v2 = st.columns([2, 1])
    with col_v1:
        fig_vib = go.Figure()
        for phase in ts_df["Phase_Label"].unique():
            mask     = ts_df["Phase_Label"] == phase
            phase_df = ts_df[mask]
            color    = phase_colors.get(phase, "#AAAAAA")
            fig_vib.add_trace(go.Scatter(
                x=phase_df["Time_min"], y=phase_df["Vibration"],
                mode="lines", name=phase,
                line=dict(color=color, width=1.5),
                hovertemplate=f"<b>{phase}</b><br>Vibration: %{{y:.3f}} mm/s<extra></extra>"
            ))

        from scipy import stats as scipy_stats
        vib_z        = np.abs(scipy_stats.zscore(ts_df["Vibration"]))
        vib_anomalies = ts_df[vib_z > 2.5]
        if not vib_anomalies.empty:
            fig_vib.add_trace(go.Scatter(
                x=vib_anomalies["Time_min"], y=vib_anomalies["Vibration"],
                mode="markers", name="⚠️ Vib Anomaly",
                marker=dict(color="red", size=9, symbol="x"),
                hovertemplate="<b>⚠️ VIBRATION SPIKE</b><br>%{y:.3f} mm/s<extra></extra>"
            ))

        fig_vib.update_layout(
            height=300, xaxis_title="Time (minutes)", yaxis_title="Vibration (mm/s)",
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=40, l=60, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="x unified"
        )
        fig_vib.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
        fig_vib.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
        st.plotly_chart(fig_vib, use_container_width=True)

    with col_v2:
        st.markdown("**Vibration Stats by Phase**")
        vib_stats = ts_df.groupby("Phase_Label")["Vibration"].agg(
            Mean="mean", Max="max", Std="std"
        ).round(3).sort_values("Max", ascending=False)
        st.dataframe(vib_stats, use_container_width=True)

    st.markdown("---")
    st.subheader("⚡ Phase Energy Efficiency Ranking")

    col_e1, col_e2 = st.columns([3, 2])
    with col_e1:
        display_phase = phase_sum[[
            "Phase_Label", "Energy_pct", "Est_Duration_min",
            "Avg_Energy", "Max_Energy", "Energy_per_min", "Anomaly_Count"
        ]].copy()
        display_phase.columns = [
            "Phase", "Energy %", "Duration (min)",
            "Avg Power (kW)", "Peak Power (kW)", "Power/min", "Anomalies"
        ]
        for col in ["Energy %", "Duration (min)", "Avg Power (kW)",
                    "Peak Power (kW)", "Power/min"]:
            display_phase[col] = display_phase[col].round(2)
        display_phase["Anomalies"] = display_phase["Anomalies"].astype(int)
        st.dataframe(display_phase, use_container_width=True, hide_index=True)

    with col_e2:
        fig_pie = go.Figure(go.Pie(
            labels=phase_sum["Phase_Label"],
            values=phase_sum["Total_Energy"],
            hole=0.45,
            marker_colors=[phase_colors.get(p, "#AAAAAA") for p in phase_sum["Phase_Label"]],
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>%{percent}<br>%{value:.1f} kW total<extra></extra>"
        ))
        fig_pie.update_layout(
            height=300, showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            annotations=[dict(text="T001<br>Energy", x=0.5, y=0.5,
                              font_size=12, showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    if n_anomalies > 0:
        st.markdown("---")
        st.subheader("🔍 Anomaly Detail Report")
        anomaly_detail = ts_df[ts_df["Is_Anomaly"]][[
            "Time_min", "Phase_Label", "Energy_kW", "Z_Score", "Vibration"
        ]].copy()
        anomaly_detail.columns = ["Time (min)", "Phase", "Power (kW)", "Z-Score", "Vibration (mm/s)"]
        anomaly_detail = anomaly_detail.round(3).sort_values("Z-Score", ascending=False)
        st.dataframe(anomaly_detail, use_container_width=True, hide_index=True)
        st.caption("Anomalies = Z-score > 2.0 within each phase.")

# =============================================================================
# PAGE 8 — HISTORY & LEARNING
# =============================================================================
elif page == "📈 History & Learning":
    st.header("📈 History & Continuous Learning")

    tab1, tab2, tab3 = st.tabs([
        "HITL Decision Log",
        "GS Evolution",
        "Batch Performance Trend"
    ])

    with tab1:
        st.subheader("Human-in-the-Loop Decision History")
        hitl_history = get_hitl_history()
        if hitl_history:
            hitl_df = pd.DataFrame(hitl_history)
            st.dataframe(hitl_df, use_container_width=True)

            decision_counts = hitl_df["decision"].value_counts()
            fig = px.pie(values=decision_counts.values,
                         names=decision_counts.index,
                         color_discrete_sequence=["#4CAF50", "#f44336", "#FF9800"])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No HITL decisions recorded yet. "
                    "Use the HITL Workflow page to make decisions.")

    with tab2:
        st.subheader("Golden Signature Evolution")
        gs_store_c = load_golden_signatures()

        for key, gs in gs_store_c.items():
            history = gs.get("update_history", [])
            current_score = float(gs["composite_score"])
            version       = gs["version"]

            with st.expander(
                f"{'🏆' if version > 1 else '📌'} {gs['scenario_label']} "
                f"— v{version} · Score: {current_score:.4f}",
                expanded=True
            ):
                # --- KPI row ---
                ck1, ck2, ck3, ck4 = st.columns(4)
                ck1.metric("Current Version", f"v{version}")
                ck2.metric("Current Score",   f"{current_score:.4f}")
                ck3.metric("Updates Applied", f"{len(history)}")

                if history:
                    first_score = float(history[0].get(
                        "old_composite_score",
                        history[0].get("composite_score", current_score)
                    ))
                    total_improvement = (
                        (current_score - first_score) / max(first_score, 1e-9) * 100
                    )
                    ck4.metric(
                        "Total Improvement",
                        f"{total_improvement:+.2f}%",
                        delta=f"since v1",
                        delta_color="normal"
                    )
                else:
                    ck4.metric("Total Improvement", "—", delta="no updates yet", delta_color="off")

                st.markdown("---")

                if history:
                    # Build history table
                    hist_rows = []
                    for h in history:
                        hist_rows.append({
                            "Timestamp"      : h.get("timestamp", "—")[:19],
                            "Old Version"    : f"v{h.get('old_version', '?')}",
                            "New Version"    : f"v{h.get('new_version', '?')}",
                            "Old Score"      : round(float(h.get("old_composite_score", 0)), 4),
                            "New Score"      : round(float(h.get("new_composite_score",
                                                    h.get("composite_score", 0))), 4),
                            "Improvement %"  : f"+{h.get('improvement_pct', 0):.2f}%",
                            "Source Batch"   : h.get("new_batch_id",
                                                h.get("source_batch_id", "—")),
                        })
                    hist_df = pd.DataFrame(hist_rows)
                    st.markdown("**Version History:**")
                    st.dataframe(hist_df, use_container_width=True, hide_index=True)

                    # Score trend chart
                    if len(hist_rows) >= 1:
                        # Build full score timeline including v1 start and current
                        timeline_scores   = []
                        timeline_versions = []

                        # Add initial score (before first update)
                        timeline_scores.append(hist_rows[0]["Old Score"])
                        timeline_versions.append(f"v{history[0].get('old_version', 1)}")

                        # Add each update
                        for row in hist_rows:
                            timeline_scores.append(row["New Score"])
                            timeline_versions.append(row["New Version"])

                        fig_trend = go.Figure()
                        fig_trend.add_trace(go.Scatter(
                            x=timeline_versions,
                            y=timeline_scores,
                            mode="lines+markers",
                            name="GS Score",
                            line=dict(color="#00D4AA", width=3),
                            marker=dict(size=10, color="#00D4AA",
                                       line=dict(color="#FFFFFF", width=2)),
                            hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>"
                        ))
                        fig_trend.update_layout(
                            height=220,
                            xaxis_title="Version",
                            yaxis_title="Composite Score",
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            margin=dict(t=10, b=30, l=60, r=20),
                        )
                        fig_trend.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
                        fig_trend.update_yaxes(showgrid=True, gridcolor="#E2E8F0",
                                               rangemode="tozero")
                        st.markdown("**Score Trend:**")
                        st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info(
                        f"**{gs['scenario_label']}** is at initial version v1. "
                        f"Use the HITL Workflow page to submit a batch that beats "
                        f"the current score of {current_score:.4f} to trigger an update."
                    )

    with tab3:
        st.subheader("Batch Performance Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=full_df["Batch_ID"], y=full_df["Quality_Score"],
            mode="lines+markers", name="Quality Score",
            line=dict(color="#2196F3")
        ))
        fig.add_trace(go.Scatter(
            x=full_df["Batch_ID"], y=full_df["Est_Total_Energy_kWh"],
            mode="lines+markers", name="Energy (kWh)",
            line=dict(color="#FF9800"), yaxis="y2"
        ))
        fig.update_layout(
            height=400,
            yaxis=dict(title="Quality Score"),
            yaxis2=dict(title="Energy (kWh)", overlaying="y", side="right"),
            xaxis=dict(tickangle=45)
        )
        st.plotly_chart(fig, use_container_width=True)
        # =============================================================================
# BACKWARDS COMPATIBILITY ALIAS
# update_golden_signature is called by the API layer
# It wraps evaluate_for_gs_update with a simpler interface
# =============================================================================
def update_golden_signature(scenario_key, new_batch_data,
                            new_composite_score, hitl_accepted=True):
    """
    Compatibility wrapper used by api/main.py.
    Delegates to evaluate_for_gs_update.
    """
    import pandas as pd
    df = pd.read_csv(config.PROCESSED_DATA_FILE)

    decision = "ACCEPT" if hitl_accepted else "REJECT"

    new_batch_metrics = {
        "batch_id"         : new_batch_data.get("batch_id", "UNKNOWN"),
        "quality_score"    : new_batch_data.get("quality_score", 0),
        "yield_score"      : new_batch_data.get("yield_score", 0),
        "performance_score": new_batch_data.get("performance_score", 0),
        "energy_kwh"       : new_batch_data.get("energy_kwh", 0),
        "carbon_kg"        : new_batch_data.get("carbon_kg", 0),
        "process_params"   : new_batch_data.get("process_params", {}),
    }

    result = evaluate_for_gs_update(
        scenario_key=scenario_key,
        new_batch_metrics=new_batch_metrics,
        reference_df=df,
        hitl_decision=decision
    )

    return result["updated"], result["message"]