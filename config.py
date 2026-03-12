# =============================================================================
# config.py — Central Configuration for Adaptive Optimization Engine
# =============================================================================

import os

# -----------------------------------------------------------------------------
# PROJECT PATHS
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW_DIR        = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")
OUTPUTS_DIR         = os.path.join(BASE_DIR, "outputs")
CHARTS_DIR          = os.path.join(BASE_DIR, "outputs", "charts")
REPORTS_DIR         = os.path.join(BASE_DIR, "outputs", "reports")

# -----------------------------------------------------------------------------
# DATASET FILENAMES
# -----------------------------------------------------------------------------
PROCESS_DATA_FILE    = os.path.join(DATA_RAW_DIR, "_h_batch_process_data.xlsx")
PRODUCTION_DATA_FILE = os.path.join(DATA_RAW_DIR, "_h_batch_production_data.xlsx")
PROCESSED_DATA_FILE  = os.path.join(DATA_PROCESSED_DIR, "batch_features.csv")
GOLDEN_SIGNATURE_FILE = os.path.join(DATA_PROCESSED_DIR, "golden_signatures.json")
HITL_LOG_FILE        = os.path.join(DATA_PROCESSED_DIR, "hitl_decisions.json")
HISTORY_LOG_FILE     = os.path.join(DATA_PROCESSED_DIR, "batch_history.json")

# -----------------------------------------------------------------------------
# ENERGY & CARBON CONSTANTS
# -----------------------------------------------------------------------------
# India grid average emission factor (kg CO2 per kWh)
CARBON_EMISSION_FACTOR = 0.000233

# T001 reference values for energy estimation (calibration batch)
T001_TOTAL_ENERGY    = 4604.28   # kW·min
T001_GRANULATION_TIME = 15       # minutes
T001_DRYING_TEMP     = 60        # °C
T001_DRYING_TIME     = 25        # minutes
T001_COMPRESSION_FORCE = 12.5   # kN
T001_MACHINE_SPEED   = 150       # RPM

# Phase energy proportions (derived from T001 time-series analysis)
PHASE_ENERGY_PROPORTIONS = {
    "Compression"     : 0.5042,
    "Drying"          : 0.1314,
    "Milling"         : 0.1173,
    "Coating"         : 0.0877,
    "Blending"        : 0.0598,
    "Granulation"     : 0.0493,
    "Quality_Testing" : 0.0384,
    "Preparation"     : 0.0119,
}

# -----------------------------------------------------------------------------
# QUALITY COMPOSITE SCORE WEIGHTS
# -----------------------------------------------------------------------------
# These define how we combine multiple quality columns into one score
QUALITY_WEIGHTS = {
    "Dissolution_Rate"    : 0.35,   # most important — drug release
    "Content_Uniformity"  : 0.25,   # consistency across tablets
    "Hardness"            : 0.20,   # structural integrity
    "Friability"          : 0.10,   # lower is better (inverted)
    "Disintegration_Time" : 0.10,   # lower is better (inverted)
}

# -----------------------------------------------------------------------------
# MULTI-OBJECTIVE OPTIMIZATION WEIGHTS
# Three Golden Signature scenarios
# -----------------------------------------------------------------------------
OPTIMIZATION_SCENARIOS = {
    "GS1_MaxQuality_MinEnergy": {
        "label"       : "Max Quality + Min Energy",
        "w_quality"   : 0.40,
        "w_yield"     : 0.15,
        "w_performance": 0.10,
        "w_energy"    : 0.25,
        "w_carbon"    : 0.10,
    },
    "GS2_MaxYield_MinCarbon": {
        "label"       : "Max Yield + Min Carbon",
        "w_quality"   : 0.15,
        "w_yield"     : 0.40,
        "w_performance": 0.10,
        "w_energy"    : 0.10,
        "w_carbon"    : 0.25,
    },
    "GS3_Balanced": {
        "label"       : "Balanced — All Objectives",
        "w_quality"   : 0.20,
        "w_yield"     : 0.20,
        "w_performance": 0.20,
        "w_energy"    : 0.20,
        "w_carbon"    : 0.20,
    },
}

# -----------------------------------------------------------------------------
# ADAPTIVE TARGET SETTINGS
# -----------------------------------------------------------------------------
# How much better than current average must a batch be to update Golden Signature
GS_UPDATE_THRESHOLD  = 0.01    # 1% improvement required minimum

# Regulatory carbon reduction target (% reduction from baseline per year)
CARBON_REDUCTION_TARGET_PCT = 0.10   # 10% annual reduction goal

# Energy reduction target per batch vs baseline
ENERGY_REDUCTION_TARGET_PCT = 0.08   # 8% reduction goal

# -----------------------------------------------------------------------------
# PROCESS PARAMETER COLUMNS
# (inputs to the optimization engine)
# -----------------------------------------------------------------------------
PROCESS_PARAM_COLS = [
    "Granulation_Time",
    "Binder_Amount",
    "Drying_Temp",
    "Drying_Time",
    "Compression_Force",
    "Machine_Speed",
    "Lubricant_Conc",
    "Moisture_Content",
]

# -----------------------------------------------------------------------------
# QUALITY OUTPUT COLUMNS
# (targets / outcomes)
# -----------------------------------------------------------------------------
QUALITY_OUTPUT_COLS = [
    "Dissolution_Rate",
    "Content_Uniformity",
    "Hardness",
    "Friability",
    "Disintegration_Time",
    "Tablet_Weight",
]