# =============================================================================
# src/data_pipeline.py — Data Loading, Cleaning & Energy Estimation
# =============================================================================

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_raw_data():
    """Load both raw datasets from the data/raw directory."""
    print("[1/5] Loading raw datasets...")

    process_df = pd.read_excel(config.PROCESS_DATA_FILE)
    production_df = pd.read_excel(config.PRODUCTION_DATA_FILE)

    print(f"      Process data    : {process_df.shape[0]} rows, {process_df.shape[1]} columns")
    print(f"      Production data : {production_df.shape[0]} rows, {production_df.shape[1]} columns")

    return process_df, production_df


def estimate_energy_for_all_batches(production_df):
    """
    Estimate energy consumption for all 60 batches.
    
    We only have detailed time-series energy data for T001.
    We use T001 as a calibration reference to estimate energy
    for all 60 batches based on their process parameters.
    
    Logic:
        - Compression energy  ∝ Compression_Force × Machine_Speed
        - Drying energy       ∝ Drying_Time × Drying_Temp
        - Granulation energy  ∝ Granulation_Time
        - Total = sum of phase estimates
    """
    print("[2/5] Estimating energy for all 60 batches...")

    df = production_df.copy()

    # Reference values from T001 (our calibration batch)
    ref_gran    = config.T001_GRANULATION_TIME
    ref_dtemp   = config.T001_DRYING_TEMP
    ref_dtime   = config.T001_DRYING_TIME
    ref_comp    = config.T001_COMPRESSION_FORCE
    ref_speed   = config.T001_MACHINE_SPEED
    total_e     = config.T001_TOTAL_ENERGY

    props = config.PHASE_ENERGY_PROPORTIONS

    # Estimate energy per phase for each batch
    df["Energy_Granulation"] = (
        (df["Granulation_Time"] / ref_gran) *
        props["Granulation"] * total_e
    )

    df["Energy_Drying"] = (
        (df["Drying_Time"] / ref_dtime) *
        (df["Drying_Temp"] / ref_dtemp) *
        props["Drying"] * total_e
    )

    df["Energy_Compression"] = (
        (df["Compression_Force"] / ref_comp) *
        (df["Machine_Speed"] / ref_speed) *
        props["Compression"] * total_e
    )

    # Remaining phases scaled by machine speed (proxy for activity level)
    df["Energy_Other"] = (
        (df["Machine_Speed"] / ref_speed) *
        (props["Milling"] + props["Coating"] +
         props["Blending"] + props["Quality_Testing"] +
         props["Preparation"]) * total_e
    )

    # Total estimated energy per batch (kW·min)
    df["Est_Total_Energy_kWmin"] = (
        df["Energy_Granulation"] +
        df["Energy_Drying"] +
        df["Energy_Compression"] +
        df["Energy_Other"]
    )

    # Convert to kWh for standard reporting
    df["Est_Total_Energy_kWh"] = df["Est_Total_Energy_kWmin"] / 60.0

    # Carbon emissions estimate (kg CO2)
    df["Est_Carbon_kg"] = (
        df["Est_Total_Energy_kWh"] * config.CARBON_EMISSION_FACTOR * 1000
    )

    print(f"      Energy range    : {df['Est_Total_Energy_kWh'].min():.1f} — {df['Est_Total_Energy_kWh'].max():.1f} kWh")
    print(f"      Carbon range    : {df['Est_Carbon_kg'].min():.3f} — {df['Est_Carbon_kg'].max():.3f} kg CO2")

    return df


def compute_quality_composite(df):
    """
    Combine multiple quality columns into one composite Quality Score (0-100).
    
    Higher is always better.
    For inverse metrics (Friability, Disintegration_Time),
    we invert them before scoring so the logic stays consistent.
    """
    print("[3/5] Computing quality composite score...")

    df = df.copy()

    # Normalize each quality column to 0-1 range
    def normalize(series, invert=False):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        normalized = (series - min_val) / (max_val - min_val)
        if invert:
            normalized = 1 - normalized
        return normalized

    w = config.QUALITY_WEIGHTS

    df["Quality_Score"] = (
        w["Dissolution_Rate"]   * normalize(df["Dissolution_Rate"])   +
        w["Content_Uniformity"] * normalize(df["Content_Uniformity"]) +
        w["Hardness"]           * normalize(df["Hardness"])           +
        w["Friability"]         * normalize(df["Friability"], invert=True) +
        w["Disintegration_Time"]* normalize(df["Disintegration_Time"], invert=True)
    ) * 100

    print(f"      Quality Score range: {df['Quality_Score'].min():.1f} — {df['Quality_Score'].max():.1f}")
    # Normalize to 0-100 scale for display clarity
    q_min = df["Quality_Score"].min()
    q_max = df["Quality_Score"].max()
    df["Quality_Score_Raw"] = df["Quality_Score"].copy()
    df["Quality_Score"] = (
        (df["Quality_Score"] - q_min) / (q_max - q_min) * 55 + 45
    ).round(2)

    return df


def compute_yield_and_performance(df):
    """
    Derive Yield and Performance scores from available process data.

    Yield proxy     : Tablet_Weight consistency (closer to target = better yield)
    Performance proxy: inverse of Moisture_Content deviation
                       (optimal moisture = better process performance)
    """
    print("[4/5] Computing yield and performance scores...")

    df = df.copy()

    # Yield Score — based on tablet weight consistency
    # Target weight assumed to be the mean; less deviation = higher yield
    target_weight = df["Tablet_Weight"].mean()
    weight_deviation = (df["Tablet_Weight"] - target_weight).abs()
    max_dev = weight_deviation.max()
    if max_dev > 0:
        df["Yield_Score"] = (1 - (weight_deviation / max_dev)) * 100
    else:
        df["Yield_Score"] = 100.0

    # P7 fix — floor at 5.0 to prevent unrealistic 0.0 values
    df["Yield_Score"] = df["Yield_Score"].clip(lower=5.0).round(2)

    # Performance Score — based on moisture content
    # Optimal moisture ~2.0%; deviation in either direction = worse performance
    optimal_moisture = 2.0
    moisture_deviation = (df["Moisture_Content"] - optimal_moisture).abs()
    max_mdev = moisture_deviation.max()
    if max_mdev > 0:
        df["Performance_Score"] = (1 - (moisture_deviation / max_mdev)) * 100
    else:
        df["Performance_Score"] = 100.0

    # Floor performance score at 5.0 as well for consistency
    df["Performance_Score"] = df["Performance_Score"].clip(lower=5.0).round(2)

    print(f"      Yield Score range      : {df['Yield_Score'].min():.1f} — {df['Yield_Score'].max():.1f}")
    print(f"      Performance Score range: {df['Performance_Score'].min():.1f} — {df['Performance_Score'].max():.1f}")

    return df

def save_processed_data(df):
    """Save the final processed dataset to data/processed/"""
    print("[5/5] Saving processed dataset...")

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)

    # --- Engineered features (P5 fix) ---
    from src.feature_engineering import build_engineered_features
    df = build_engineered_features(df)
    print(f"[Pipeline] Engineered features added. Final shape: {df.shape}")

    df.to_csv(config.PROCESSED_DATA_FILE, index=False)
    print(f"      Saved to : {config.PROCESSED_DATA_FILE}")
    print(f"      Shape    : {df.shape[0]} rows × {df.shape[1]} columns")

    return df


def run_pipeline():
    """
    Master function — runs the complete data pipeline end to end.
    Returns the final processed DataFrame with all engineered features.
    """
    print("=" * 60)
    print("  ADAPTIVE OPTIMIZATION ENGINE — Data Pipeline")
    print("=" * 60)

    # Step 1: Load
    process_df, production_df = load_raw_data()

    # Step 2: Energy estimation
    df = estimate_energy_for_all_batches(production_df)

    # Step 3: Quality composite
    df = compute_quality_composite(df)

    # Step 4: Yield + Performance
    df = compute_yield_and_performance(df)

    # Step 5: Save — now returns enriched df
    df = save_processed_data(df)

    print("=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)

    return df


if __name__ == "__main__":
    df = run_pipeline()
    print("\nFinal columns:")
    print(df.columns.tolist())
    print("\nSample output:")
    print(df[["Batch_ID", "Quality_Score", "Yield_Score",
              "Performance_Score", "Est_Total_Energy_kWh",
              "Est_Carbon_kg"]].head(10))
    
# =============================================================================
# T001 TIME-SERIES LOADER & ANOMALY DETECTION
# =============================================================================

def load_timeseries_data():
    """
    Load and enrich the T001 batch process time-series dataset.
    (_h_batch_process_data.xlsx — 211 rows, 8 phases)

    Enrichments added:
    - Cumulative time index
    - Rolling mean energy (window=5)
    - Z-score per phase for anomaly detection
    - Anomaly flag (|z| > 2.0)
    - Phase energy efficiency (energy per minute)

    Returns:
        pd.DataFrame with all enrichments, or None if file not found
    """
    import scipy.stats as stats

    filepath = os.path.join(config.DATA_RAW_DIR, "_h_batch_process_data.xlsx")

    if not os.path.exists(filepath):
        print(f"[TS] Time-series file not found: {filepath}")
        return None

    print(f"[TS] Loading time-series data from {filepath}...")
    df = pd.read_excel(filepath)
    print(f"[TS] Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[TS] Columns: {list(df.columns)}")

    # Identify the energy column (flexible — handles different column names)
    energy_col = None
    for candidate in ["Power_Consumption_kW", "Energy_kW", "Energy",
                      "Power_kW", "Power", "energy_kw", "energy", "kW", "KW"]:
        if candidate in df.columns:
            energy_col = candidate
            break

    # If no exact match, take the first numeric column that isn't time/index
    if energy_col is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        non_phase = [c for c in numeric_cols if "phase" not in c.lower()
                     and "time" not in c.lower() and "step" not in c.lower()]
        if non_phase:
            energy_col = non_phase[0]

    print(f"[TS] Using energy column: {energy_col}")

    # Identify phase column
    phase_col = None
    for candidate in ["Phase", "phase", "Stage", "stage",
                      "Process_Phase", "process_phase"]:
        if candidate in df.columns:
            phase_col = candidate
            break

    if phase_col is None:
        # Take first string/object column as phase
        obj_cols = df.select_dtypes(include="object").columns.tolist()
        if obj_cols:
            phase_col = obj_cols[0]

    print(f"[TS] Using phase column: {phase_col}")

    # Identify time column
    time_col = None
    for candidate in ["Time", "time", "Timestamp", "timestamp",
                      "Time_min", "time_min", "Minutes", "Step"]:
        if candidate in df.columns:
            time_col = candidate
            break

    df = df.copy()

    # --- Cumulative row index as time proxy if no time column ---
    if time_col:
        df["Time_Index"] = df[time_col]
    else:
        df["Time_Index"] = range(len(df))

    # --- Rename for consistency ---
    df["Energy_kW"]    = df[energy_col] if energy_col else 0
    df["Phase_Label"]  = df[phase_col]  if phase_col  else "Unknown"
    df["Vibration"]    = df["Vibration_mm_s"] if "Vibration_mm_s" in df.columns else 0
    df["Time_min"]     = df["Time_Minutes"]   if "Time_Minutes"   in df.columns else df["Time_Index"]

    # --- Rolling mean (smoothed energy trend) ---
    df["Energy_Rolling"] = df["Energy_kW"].rolling(window=5, center=True).mean()
    df["Energy_Rolling"].fillna(df["Energy_kW"], inplace=True)

    # --- Z-score anomaly detection per phase ---
    df["Z_Score"]   = 0.0
    df["Is_Anomaly"] = False

    for phase in df["Phase_Label"].unique():
        mask = df["Phase_Label"] == phase
        phase_data = df.loc[mask, "Energy_kW"]

        if len(phase_data) > 2:
            z = np.abs(stats.zscore(phase_data))
            df.loc[mask, "Z_Score"]    = z
            df.loc[mask, "Is_Anomaly"] = z > 2.0

    # --- Phase summary ---
    phase_summary = (
        df.groupby("Phase_Label")
        .agg(
            Row_Count    = ("Energy_kW", "count"),
            Avg_Energy   = ("Energy_kW", "mean"),
            Max_Energy   = ("Energy_kW", "max"),
            Min_Energy   = ("Energy_kW", "min"),
            Total_Energy = ("Energy_kW", "sum"),
            Anomaly_Count= ("Is_Anomaly", "sum"),
        )
        .reset_index()
    )

    # Estimate duration in minutes (assume each row = 1 observation interval)
    # T001 total time is approximately 211 minutes
    minutes_per_row = 211.0 / len(df)
    phase_summary["Est_Duration_min"] = phase_summary["Row_Count"] * minutes_per_row
    phase_summary["Energy_per_min"]   = (
        phase_summary["Total_Energy"] / phase_summary["Est_Duration_min"].replace(0, 1)
    )
    phase_summary["Energy_pct"] = (
        phase_summary["Total_Energy"] / phase_summary["Total_Energy"].sum() * 100
    ).round(1)
    phase_summary = phase_summary.sort_values("Total_Energy", ascending=False)

    n_anomalies = int(df["Is_Anomaly"].sum())
    print(f"[TS] Anomalies detected: {n_anomalies} / {len(df)} points")
    print(f"[TS] Phases found: {df['Phase_Label'].nunique()}")

    return {
        "timeseries"    : df,
        "phase_summary" : phase_summary,
        "energy_col"    : energy_col,
        "phase_col"     : phase_col,
        "time_col"      : time_col,
        "n_anomalies"   : n_anomalies,
        "n_points"      : len(df),
    }