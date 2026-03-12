# =============================================================================
# src/feature_engineering.py — ML Models: Quality & Energy Prediction
# =============================================================================

import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import pickle


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_engineered_features(df):
    """
    Build additional engineered features from raw process parameters.
    Safe to call multiple times — checks before adding columns.
    Handles missing upstream columns gracefully.
    """
    df = df.copy()

    # Guard: only compute if required source columns exist
    has_energy   = "Est_Total_Energy_kWh" in df.columns
    has_quality  = "Quality_Score"        in df.columns
    has_yield    = "Yield_Score"          in df.columns
    has_carbon   = "Est_Carbon_kg"        in df.columns

    # --- Energy per Quality Unit (lower = more efficient) ---
    if has_energy and has_quality:
        df["Energy_per_Quality"] = (
            df["Est_Total_Energy_kWh"] / df["Quality_Score"].replace(0, 1)
        ).round(4)

    # --- Overall Efficiency Ratio (higher = better) ---
    if has_energy and has_quality and has_yield:
        df["Efficiency_Ratio"] = (
            df["Quality_Score"] * df["Yield_Score"]
        ) / (df["Est_Total_Energy_kWh"].replace(0, 1))
        df["Efficiency_Ratio"] = df["Efficiency_Ratio"].round(4)

    # --- Carbon per Quality Unit ---
    if has_carbon and has_quality:
        df["Carbon_per_Quality"] = (
            df["Est_Carbon_kg"] / df["Quality_Score"].replace(0, 1)
        ).round(4)

    # --- Process parameter interactions ---
    if "Drying_Temp" in df.columns and "Drying_Time" in df.columns:
        df["Drying_Intensity"] = (
            df["Drying_Temp"] * df["Drying_Time"]
        ).round(2)

    if "Compression_Force" in df.columns and "Machine_Speed" in df.columns:
        df["Compression_Intensity"] = (
            df["Compression_Force"] * df["Machine_Speed"]
        ).round(2)

    if "Moisture_Content" in df.columns:
        df["Moisture_Deviation"] = (
            (df["Moisture_Content"] - 2.0).abs()
        ).round(4)

    if "Binder_Amount" in df.columns and "Granulation_Time" in df.columns:
        df["Granulation_Efficiency"] = (
            df["Binder_Amount"] / df["Granulation_Time"].replace(0, 1)
        ).round(4)

    # --- Phase energy imbalance (compression dominance ratio) ---
    if "Energy_Compression" in df.columns and "Est_Total_Energy_kWmin" in df.columns:
        df["Compression_Energy_Ratio"] = (
            df["Energy_Compression"] / df["Est_Total_Energy_kWmin"].replace(0, 1)
        ).round(4)

    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================

FEATURE_COLS = config.PROCESS_PARAM_COLS + [
    "Drying_Intensity",
    "Compression_Intensity",
    "Moisture_Deviation",
    "Granulation_Efficiency",
]

MODEL_DIR = config.DATA_PROCESSED_DIR


def train_quality_model(df):
    """
    Train XGBoost to predict Dissolution_Rate (raw quality measurement).
    Dissolution_Rate has 0.98+ correlations with process params and
    full natural variance — far better for ML than derived Quality_Score.
    We convert predictions back to Quality_Score scale for display.
    """
    print("[ML] Training Quality (Dissolution_Rate) prediction model...")

    QUALITY_FEATURES = [
        "Granulation_Time",
        "Binder_Amount",
        "Drying_Time",
        "Machine_Speed",
        "Moisture_Content",
        "Compression_Force",
        "Lubricant_Conc",
    ]

    X = df[QUALITY_FEATURES].values
    y = df["Dissolution_Rate"].values  # raw measurement — full variance

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler.feature_names = QUALITY_FEATURES

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )

    kf     = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2  = cross_val_score(model, X_scaled, y, cv=kf, scoring="r2")
    cv_mse = cross_val_score(model, X_scaled, y, cv=kf,
                             scoring="neg_mean_squared_error")

    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    metrics = {
        "model"           : "XGBoost Regressor",
        "target"          : "Dissolution_Rate",
        "target_display"  : "Quality Score (via Dissolution Rate)",
        "n_samples"       : len(y),
        "n_features"      : len(QUALITY_FEATURES),
        "r2_train"        : round(r2_score(y, y_pred), 4),
        "r2_cv_mean"      : round(cv_r2.mean(), 4),
        "r2_cv_std"       : round(cv_r2.std(), 4),
        "rmse_train"      : round(np.sqrt(mean_squared_error(y, y_pred)), 4),
        "mae_train"       : round(mean_absolute_error(y, y_pred), 4),
        "rmse_cv_mean"    : round(np.sqrt((-cv_mse).mean()), 4),
        "feature_names"   : QUALITY_FEATURES,
        "dissolution_mean": round(float(df["Dissolution_Rate"].mean()), 4),
        "dissolution_std" : round(float(df["Dissolution_Rate"].std()), 4),
        "quality_mean"    : round(float(df["Quality_Score"].mean()), 4),
        "quality_std"     : round(float(df["Quality_Score"].std()), 4),
    }

    importance = dict(zip(QUALITY_FEATURES,
                          model.feature_importances_.round(4).tolist()))
    metrics["feature_importance"] = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)
    )

    print(f"      Target         : Dissolution_Rate (raw)")
    print(f"      R² (train)     : {metrics['r2_train']}")
    print(f"      R² (CV mean)   : {metrics['r2_cv_mean']} ± {metrics['r2_cv_std']}")
    print(f"      RMSE (train)   : {metrics['rmse_train']}")
    print(f"      MAE  (train)   : {metrics['mae_train']}")

    return model, scaler, metrics


def train_energy_model(df):
    """
    Train XGBoost model to predict Energy consumption from process parameters.
    """
    print("[ML] Training Energy prediction model...")

    df = build_engineered_features(df)

    X = df[FEATURE_COLS].values
    y = df["Est_Total_Energy_kWh"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbosity=0
    )

    kf     = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2  = cross_val_score(model, X_scaled, y, cv=kf, scoring="r2")
    cv_mse = cross_val_score(model, X_scaled, y, cv=kf,
                             scoring="neg_mean_squared_error")

    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    metrics = {
        "model"         : "XGBoost Regressor",
        "target"        : "Est_Total_Energy_kWh",
        "n_samples"     : len(y),
        "n_features"    : len(FEATURE_COLS),
        "r2_train"      : round(r2_score(y, y_pred), 4),
        "r2_cv_mean"    : round(cv_r2.mean(), 4),
        "r2_cv_std"     : round(cv_r2.std(), 4),
        "rmse_train"    : round(np.sqrt(mean_squared_error(y, y_pred)), 4),
        "mae_train"     : round(mean_absolute_error(y, y_pred), 4),
        "rmse_cv_mean"  : round(np.sqrt((-cv_mse).mean()), 4),
        "feature_names" : FEATURE_COLS,
    }

    importance = dict(zip(FEATURE_COLS,
                          model.feature_importances_.round(4).tolist()))
    metrics["feature_importance"] = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)
    )

    print(f"      R² (train)     : {metrics['r2_train']}")
    print(f"      R² (CV mean)   : {metrics['r2_cv_mean']} ± {metrics['r2_cv_std']}")
    print(f"      RMSE (train)   : {metrics['rmse_train']}")
    print(f"      MAE  (train)   : {metrics['mae_train']}")

    return model, scaler, metrics


def save_models(quality_model, quality_scaler, quality_metrics,
                energy_model,  energy_scaler,  energy_metrics):
    """Save trained models and metrics to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    pickle.dump(quality_model,
                open(os.path.join(MODEL_DIR, "quality_model.pkl"), "wb"))
    pickle.dump(quality_scaler,
                open(os.path.join(MODEL_DIR, "quality_scaler.pkl"), "wb"))
    pickle.dump(energy_model,
                open(os.path.join(MODEL_DIR, "energy_model.pkl"),  "wb"))
    pickle.dump(energy_scaler,
                open(os.path.join(MODEL_DIR, "energy_scaler.pkl"),  "wb"))

    all_metrics = {
        "quality_model": quality_metrics,
        "energy_model" : energy_metrics
    }
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"[ML] Models saved to {MODEL_DIR}")


def load_models():
    """Load trained models from disk."""
    quality_model  = pickle.load(
        open(os.path.join(MODEL_DIR, "quality_model.pkl"), "rb"))
    quality_scaler = pickle.load(
        open(os.path.join(MODEL_DIR, "quality_scaler.pkl"), "rb"))
    energy_model   = pickle.load(
        open(os.path.join(MODEL_DIR, "energy_model.pkl"),  "rb"))
    energy_scaler  = pickle.load(
        open(os.path.join(MODEL_DIR, "energy_scaler.pkl"),  "rb"))

    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "r") as f:
        metrics = json.load(f)

    return (quality_model, quality_scaler,
            energy_model,  energy_scaler, metrics)


def models_exist():
    """Check if trained models are already saved."""
    return all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in [
        "quality_model.pkl", "quality_scaler.pkl",
        "energy_model.pkl",  "energy_scaler.pkl",
        "model_metrics.json"
    ])


def predict_outcomes(process_params_dict, df_reference):
    """
    Predict Dissolution_Rate (converted to Quality Score) and Energy
    for a new batch configuration using trained XGBoost models.

    Args:
        process_params_dict : dict of process parameter values
        df_reference        : reference DataFrame (used for context only)

    Returns:
        dict with predicted_quality, predicted_energy, predicted_carbon,
                    predicted_dissolution
    """
    if not models_exist():
        return None

    quality_model, quality_scaler, energy_model, energy_scaler, saved_metrics = load_models()

    # Build feature row from input params
    row = {col: process_params_dict.get(col, 0) for col in config.PROCESS_PARAM_COLS}

    # Engineered features for energy model
    row["Drying_Intensity"]       = row["Drying_Temp"] * row["Drying_Time"]
    row["Compression_Intensity"]  = row["Compression_Force"] * row["Machine_Speed"]
    row["Moisture_Deviation"]     = abs(row["Moisture_Content"] - 2.0)
    row["Granulation_Efficiency"] = row["Binder_Amount"] / max(row["Granulation_Time"], 1)

    # -------------------------------------------------------------------------
    # QUALITY PREDICTION
    # Model predicts Dissolution_Rate, we convert to Quality_Score scale
    # -------------------------------------------------------------------------
    QUALITY_FEATURES = [
        "Granulation_Time",
        "Binder_Amount",
        "Drying_Time",
        "Machine_Speed",
        "Moisture_Content",
        "Compression_Force",
        "Lubricant_Conc",
    ]
    X_q = np.array([[row[col] for col in QUALITY_FEATURES]])

    pred_dissolution = float(quality_model.predict(
        quality_scaler.transform(X_q)
    )[0])

    # Convert dissolution rate to quality score using saved distribution stats
    qm        = saved_metrics.get("quality_model", {})
    diss_mean = qm.get("dissolution_mean", 90.93)
    diss_std  = qm.get("dissolution_std",  4.63)
    q_mean    = qm.get("quality_mean",     54.94)
    q_std     = qm.get("quality_std",      2.85)

    pred_quality = q_mean + ((pred_dissolution - diss_mean) / diss_std) * q_std

    # -------------------------------------------------------------------------
    # ENERGY PREDICTION
    # -------------------------------------------------------------------------
    X_e = np.array([[row[col] for col in FEATURE_COLS]])

    pred_energy = float(energy_model.predict(
        energy_scaler.transform(X_e)
    )[0])

    pred_carbon = pred_energy * config.CARBON_EMISSION_FACTOR * 1000

    # -------------------------------------------------------------------------
    # Clamp to realistic ranges
    # -------------------------------------------------------------------------
    pred_quality     = max(40.0,  min(75.0,  pred_quality))
    pred_energy      = max(50.0,  min(100.0, pred_energy))
    pred_carbon      = max(10.0,  min(25.0,  pred_carbon))
    pred_dissolution = max(81.2,  min(99.9,  pred_dissolution))

    return {
        "predicted_quality"     : round(pred_quality,     2),
        "predicted_energy"      : round(pred_energy,      2),
        "predicted_carbon"      : round(pred_carbon,      4),
        "predicted_dissolution" : round(pred_dissolution, 2),
    }


# =============================================================================
# MAIN — Train and save both models
# =============================================================================
if __name__ == "__main__":
    df = pd.read_csv(config.PROCESSED_DATA_FILE)

    print("=" * 60)
    print("  ML MODEL TRAINING — Quality & Energy Prediction")
    print("=" * 60)

    q_model, q_scaler, q_metrics = train_quality_model(df)
    e_model, e_scaler, e_metrics = train_energy_model(df)

    save_models(q_model, q_scaler, q_metrics,
                e_model, e_scaler, e_metrics)

    print("\n" + "=" * 60)
    print("  FINAL MODEL SUMMARY")
    print("=" * 60)
    print(f"\nQuality Model:")
    print(f"  R² Train : {q_metrics['r2_train']}")
    print(f"  R² CV    : {q_metrics['r2_cv_mean']} ± {q_metrics['r2_cv_std']}")
    print(f"  RMSE     : {q_metrics['rmse_train']}")
    print(f"\nEnergy Model:")
    print(f"  R² Train : {e_metrics['r2_train']}")
    print(f"  R² CV    : {e_metrics['r2_cv_mean']} ± {e_metrics['r2_cv_std']}")
    print(f"  RMSE     : {e_metrics['rmse_train']}")
    print(f"\nTop Quality Predictors:")
    for feat, imp in list(q_metrics["feature_importance"].items())[:5]:
        print(f"  {feat:30s}: {imp:.4f}")
    print(f"\nTop Energy Predictors:")
    for feat, imp in list(e_metrics["feature_importance"].items())[:5]:
        print(f"  {feat:30s}: {imp:.4f}")