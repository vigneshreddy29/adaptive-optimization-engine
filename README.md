# AI-Driven Adaptive Optimization Engine
### Energy-Efficient & Sustainable Manufacturing
**Team VORTEX** | YUVAAN National AI & ML Hackathon | IIT Hyderabad x AVEVA

---

## Project Overview

An end-to-end AI-powered optimization system for pharmaceutical tablet
manufacturing that continuously learns from production data, maintains
versioned Golden Signatures, sets adaptive energy and carbon targets,
and integrates Human-in-the-Loop workflows for continuous improvement.

- **Track:** Option B — Optimization Engine Specialization
- **Domain:** Pharmaceutical Tablet Manufacturing
- **Dataset:** 60 production batches x 34 engineered features
- **Lines of Code:** 2,400+
- **Dashboard Pages:** 8
- **API Endpoints:** 10

---

## Problem Statement

Pharmaceutical batch manufacturing has no feedback loop. Engineers repeat
the same parameters even when batches underperform. Energy consumption
varies from 58.6 to 91.5 kWh per batch with no systematic optimization.
Carbon emissions are never tracked at the batch level.

Our system solves this by learning from every batch, identifying the
best-ever parameters across multiple objectives, and giving operators
quantified recommendations — with human approval before any change.

---

## Key Results

| Metric | Value |
|---|---|
| Total Batches Analyzed | 60 |
| Engineered Features | 34 (25 original + 7 new + 2 derived) |
| Pareto-Optimal Batches | 21 / 60 (35%) |
| Best Energy Achieved | 58.6 kWh vs 79.8 kWh average |
| Max Energy Saving | 24.1% vs fleet average |
| Max Carbon Saving | 24.1% (14.11 vs 18.59 kg CO2) |
| Quality Model R2 (CV) | 0.9658 +/- 0.009 |
| Energy Model R2 (CV) | 0.9606 +/- 0.016 |
| GS Scenarios | 3 |
| API Endpoints | 10 |
| T001 Anomalies Detected | 8 (z-score > 2.0) |

---

## System Architecture

```
Raw Excel Data (2 files, 271 records)
        |
        v
src/data_pipeline.py         5-step pipeline, 60 x 34 features
        |
        v
src/feature_engineering.py   XGBoost training, quality + energy models
        |
        v
src/optimization_engine.py   Vectorized Pareto front, 3 scenario scoring
        |
        v
src/golden_signature.py      Versioned GS store, HITL evaluation, audit log
        |
        v
src/adaptive_targets.py      Regulatory pressure -> dynamic targets
        |
        v
src/recommendation.py        Parameter gap analysis, impact forecasting
        |
        +------------------+------------------+
        |                                     |
        v                                     v
dashboard/app.py                         api/main.py
8-page Streamlit UI                      10-endpoint FastAPI
localhost:8501                           localhost:8000/docs
```

---

## Project Structure

```
AdaptiveOptimizationEngine/
|-- data/
|   |-- raw/
|   |   |-- _h_batch_process_data.xlsx     (211 rows, T001 time-series)
|   |   +-- _h_batch_production_data.xlsx  (60 rows, all batches)
|   +-- processed/
|       |-- batch_features.csv             (60 x 34 engineered features)
|       |-- golden_signatures.json         (versioned GS store)
|       |-- hitl_decisions.json            (full audit log)
|       |-- quality_model.pkl              (XGBoost, R2 CV = 0.966)
|       |-- energy_model.pkl               (XGBoost, R2 CV = 0.961)
|       |-- quality_scaler.pkl
|       |-- energy_scaler.pkl
|       +-- model_metrics.json
|-- src/
|   |-- data_pipeline.py                   5-step pipeline
|   |-- feature_engineering.py             XGBoost training + 7 features
|   |-- optimization_engine.py             Vectorized Pareto + scoring
|   |-- golden_signature.py                GS versioning + HITL eval
|   |-- adaptive_targets.py                Regulatory pressure logic
|   +-- recommendation.py                  Gap analysis engine
|-- dashboard/
|   +-- app.py                             8-page Streamlit dashboard
|-- api/
|   +-- main.py                            FastAPI 10-endpoint REST API
|-- config.py                              Single source of truth
|-- requirements.txt
+-- README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- Windows / Linux / macOS

### Steps

```bash
# 1. Navigate to project folder
cd AdaptiveOptimizationEngine

# 2. Create virtual environment
python -m venv venv

# 3. Activate
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify datasets exist in data/raw/
#    _h_batch_process_data.xlsx
#    _h_batch_production_data.xlsx
```

---

## Running the System

### Dashboard (Recommended)
```bash
streamlit run dashboard/app.py
```
Opens at: http://localhost:8501

### FastAPI Backend
```bash
uvicorn api.main:app --reload --port 8000
```
Swagger UI at: http://localhost:8000/docs

### Run Pipeline Only
```bash
python src/data_pipeline.py
python src/optimization_engine.py
```

---

## Dashboard Pages

| Page | Description |
|---|---|
| Overview | 5 KPIs, energy histogram, quality scatter, GS summary, regulatory pressure slider |
| Data Explorer | Correlation heatmap, parameter distributions, batch scores table |
| Optimization Engine | 3D Pareto scatter, 3-scenario comparison, composite score ranking |
| Golden Signatures | Best-batch cards, parameter profiles, version history |
| HITL Workflow | 8-param sliders, GS gap analysis, ACCEPT / REJECT / REPRIORITIZE |
| Model Intelligence | R2 metrics, feature importance, live prediction widget |
| T001 Analysis | 211-point time-series, z-score anomalies, vibration profile, phase efficiency |
| History & Learning | HITL audit log, GS evolution chart, score trend |

---

## Machine Learning Models

### Quality Prediction Model
- **Algorithm:** XGBoost Regressor
- **Target:** Dissolution_Rate -> Quality Score (0-100)
- **R2 Train:** 0.9938
- **R2 CV:** 0.9658 +/- 0.009
- **RMSE:** 0.363
- **Top Feature:** Machine_Speed (32.5%)

### Energy Prediction Model
- **Algorithm:** XGBoost Regressor
- **Target:** Est_Total_Energy_kWh
- **R2 Train:** 0.9996
- **R2 CV:** 0.9606 +/- 0.016
- **RMSE:** 0.175
- **Top Feature:** Drying_Time (48.9%)

---

## Engineered Features

| Feature | Formula | Purpose |
|---|---|---|
| Energy_per_Quality | Energy / Quality_Score | Efficiency ratio |
| Efficiency_Ratio | Quality x Yield / Energy | Overall productivity per kWh |
| Carbon_per_Quality | Carbon / Quality_Score | Sustainability metric |
| Drying_Intensity | Drying_Temp x Drying_Time | Thermal exposure |
| Compression_Intensity | Compression_Force x Machine_Speed | Mechanical stress |
| Moisture_Deviation | abs(Moisture - 2.0) | Distance from optimal |
| Granulation_Efficiency | Binder_Amount / Granulation_Time | Binder utilization rate |

---

## Optimization Engine

### Multi-Objective Optimization
Five objectives scored and weighted per scenario:
- Maximize: Quality_Score, Yield_Score, Performance_Score
- Minimize: Est_Total_Energy_kWh, Est_Carbon_kg

### Vectorized Pareto Algorithm
Numpy broadcasting approach — (n,1,m) vs (1,n,m) tensor comparison.
Dominance check across 5 objectives in a single operation.
Result: 21/60 batches on the Pareto front in 4.74ms.

### Golden Signature Scenarios

| Scenario | Priority | Best Batch | Score | Energy |
|---|---|---|---|---|
| GS1 — Max Quality + Min Energy | Quality focus | T009 | 0.8214 | 72.49 kWh |
| GS2 — Max Yield + Min Carbon | Sustainability | T005 | 0.8612 | 60.54 kWh |
| GS3 — Balanced All Objectives | Equal weights | T005 | 0.7812 | 60.54 kWh |

---

## Human-in-the-Loop Workflow

```
Step 1: Operator configures 8 process parameters via sliders
        -> AI predicts Quality, Energy, Carbon instantly

Step 2: System shows gap analysis vs current Golden Signature
        -> Parameter differences and expected impact displayed

Step 3: Operator makes one of three decisions:
        ACCEPT      -> GS update evaluation runs
                       Score >= 1% better -> GS version increments
                       Full audit entry created with timestamp
        REJECT      -> Decision logged, GS unchanged
        REPRIORITIZE -> Weights adjusted, new GS identified

Step 4: All decisions persisted to hitl_decisions.json
        GS evolution visible in History & Learning page
```

### GS Update Rule
A new GS version is created only when:
1. New batch composite score exceeds current GS by >= 1%
2. Human operator explicitly clicks ACCEPT

---

## T001 Time-Series Analysis

- **Data points:** 211 across 8 production phases
- **Anomaly detection:** Z-score per phase, threshold > 2.0
- **Anomalies found:** 8 (highest: Compression, minute 141, Z=3.3, 66.1 kW)
- **Phase energy breakdown:**

| Phase | Energy Share |
|---|---|
| Compression | 50.4% |
| Drying | 13.1% |
| Milling | 11.7% |
| Coating | 8.8% |
| Blending | 6.0% |
| Granulation | 4.9% |
| Quality_Testing | 3.8% |
| Preparation | 1.2% |

---

## Adaptive Targets

The regulatory pressure slider (0.0 to 1.0) dynamically adjusts targets:

| Pressure | Mode | Energy Target | Carbon Target |
|---|---|---|---|
| 0.0 | Relaxed | 79.8 kWh | 18.59 kg |
| 0.5 | Moderate | 76.6 kWh | 17.66 kg |
| 1.0 | Stringent | 73.4 kWh | 16.73 kg |

Base targets: 8% energy reduction, 10% carbon reduction from fleet average.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Health check |
| GET | /health | System status |
| POST | /optimize | Run optimization for scenario |
| GET | /optimize/all | All 3 scenarios |
| GET | /golden-signature | All GS summary |
| GET | /golden-signature/{key} | Specific GS details |
| POST | /hitl-decision | Submit HITL decision |
| GET | /hitl-history | Full decision log |
| POST | /recommend | Get parameter recommendations |
| GET | /adaptive-targets | Current energy/carbon targets |

---

## Configuration (config.py)

All constants centralized — no hardcoded values anywhere else:

```python
CARBON_EMISSION_FACTOR = 0.000233      # India grid kg CO2/Wh
T001_TOTAL_ENERGY = 4604.28            # kW.min calibration reference
ENERGY_REDUCTION_TARGET_PCT = 0.08    # 8% energy reduction target
CARBON_REDUCTION_TARGET_PCT = 0.10    # 10% carbon reduction target
GS_UPDATE_THRESHOLD = 0.01            # 1% minimum improvement for GS update
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| pandas | 2.1.4 | Data processing |
| numpy | 1.26.4 | Vectorized computation |
| scikit-learn | 1.3.2 | Cross-validation, scaling |
| xgboost | 2.0.3 | Quality and energy prediction |
| joblib | 1.3.2 | Model serialization |
| plotly | 5.18.0 | Interactive visualizations |
| streamlit | 1.31.0 | 8-page dashboard |
| fastapi | 0.109.2 | REST API framework |
| uvicorn | 0.27.1 | ASGI server |
| scipy | 1.11.4 | Scientific computing |
| openpyxl | 3.1.2 | Excel ingestion |

---

## Real-World Scalability

- **AVEVA Integration:** FastAPI layer accepts batch data from any AVEVA PI
  or SCADA system via POST /optimize
- **Multi-product:** config.py changes alone adapt the engine to capsules,
  injectables, food manufacturing, or any batch process
- **Edge-ready:** JSON store + CSV pipeline — no database dependency
- **Carbon tracking:** India grid factor (0.233 kg/kWh) — swap for any region

---

## Team

**Team VORTEX**
National AI & ML Hackathon — YUVAAN
Organized by Tinkerers' Lab, IIT Hyderabad
Powered by AVEVA

---

*Built with precision. Optimized for impact.*
