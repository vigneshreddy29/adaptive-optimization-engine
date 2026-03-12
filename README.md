# ⚡ AI-Driven Adaptive Optimization Engine
### Energy-Efficient & Sustainable Manufacturing
**Team VORTEX** | YUVAAN National AI & ML Hackathon | IIT Hyderabad × AVEVA

---

## 🏆 Project Overview

An end-to-end AI-powered optimization system for pharmaceutical tablet
manufacturing that dynamically manages Golden Signatures, sets adaptive
energy and carbon targets, and integrates Human-in-the-Loop workflows
to drive continuous improvement in sustainable manufacturing.

**Track:** Option B — Optimization Engine Specialization
**Domain:** Pharmaceutical Tablet Manufacturing
**Dataset:** 60 production batches × 15 process parameters

---

## 🎯 Problem Statement Addressed

| Requirement | Implementation |
|---|---|
| Golden Signature Management | 3-scenario GS store with versioning, dominance checks, update history |
| HITL Workflows | ACCEPT / REJECT / REPRIORITIZE decisions with full audit log |
| Adaptive Target Setting | Dynamic energy & carbon goals calibrated to batch history |
| Continuous Improvement | Pareto dominance check triggers GS updates automatically |
| Decision Support System | Real-time parameter gap analysis & impact forecasting |
| Integration APIs | 10 FastAPI endpoints with Swagger documentation |
| Data Processing Pipeline | T001-calibrated energy estimation across all 60 batches |

---

## 🧠 System Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                           │
│  T001 time-series → Phase energy profiles               │
│  60-batch production data → Feature engineering         │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              OPTIMIZATION ENGINE                        │
│  • Multi-objective scoring (5 objectives)               │
│  • Pareto front identification (21/60 batches)          │
│  • 3 Golden Signature scenarios                         │
│  • Weighted composite scoring                           │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│           GOLDEN SIGNATURE STORE                        │
│  GS1: Max Quality + Min Energy   → Batch T009 (0.8229)  │
│  GS2: Max Yield + Min Carbon     → Batch T005 (0.8652)  │
│  GS3: Balanced All Objectives    → Batch T005 (0.7885)  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│         HUMAN-IN-THE-LOOP INTERFACE                     │
│  • Interactive parameter configuration                  │
│  • Real-time GS comparison & gap analysis               │
│  • ACCEPT / REJECT / REPRIORITIZE decisions             │
│  • Full decision audit log                              │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│            INTEGRATION APIs (FastAPI)                   │
│  POST /optimize  •  GET /golden-signature/{scenario}    │
│  POST /hitl-decision  •  GET /adaptive-targets          │
│  POST /recommend  •  GET /history                       │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Total Batches Analyzed | 60 |
| Pareto-Optimal Batches | 21 / 60 (35%) |
| Best Energy Achieved | 58.6 kWh (vs 79.8 avg) |
| Best Carbon Achieved | 13.66 kg CO₂ (vs 18.59 avg) |
| Max Energy Saving vs Avg | 24.1% |
| Max Carbon Saving vs Avg | 24.1% |
| Max Quality Score | 61.25 / 100 |
| GS Scenarios | 3 |
| API Endpoints | 10 |

---

## 🗂️ Project Structure
```
AdaptiveOptimizationEngine/
├── data/
│   ├── raw/                        # Original datasets
│   │   ├── _h_batch_process_data.xlsx
│   │   └── _h_batch_production_data.xlsx
│   └── processed/                  # Generated outputs
│       ├── batch_features.csv      # 60 batches × 25 features
│       ├── golden_signatures.json  # GS store
│       └── hitl_decisions.json     # HITL audit log
├── src/
│   ├── data_pipeline.py            # Data loading & energy estimation
│   ├── feature_engineering.py      # Feature computation
│   ├── optimization_engine.py      # Pareto + multi-objective scoring
│   ├── golden_signature.py         # GS store, update logic, HITL log
│   ├── adaptive_targets.py         # Dynamic energy/carbon targets
│   └── recommendation.py          # Parameter adjustment engine
├── dashboard/
│   └── app.py                      # Streamlit 6-page dashboard
├── api/
│   └── main.py                     # FastAPI 10-endpoint REST API
├── config.py                       # Central configuration
├── requirements.txt                # Pinned dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.11+
- Windows / Linux / macOS

### Steps
```bash
# 1. Clone / extract project
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

# 5. Place datasets into data/raw/
#    _h_batch_process_data.xlsx
#    _h_batch_production_data.xlsx
```

---

## 🚀 Running the System

### Option A — Streamlit Dashboard (Recommended for Demo)
```bash
streamlit run dashboard/app.py
```
Opens at: http://localhost:8501

### Option B — FastAPI Backend
```bash
uvicorn api.main:app --reload --port 8000
```
Swagger UI at: http://localhost:8000/docs

### Option C — Run Pipeline Only
```bash
python src/data_pipeline.py
python src/optimization_engine.py
python src/golden_signature.py
```

---

## 🔬 Core Algorithms

### Multi-Objective Optimization
Five objectives scored and weighted per scenario:
- **Quality Score** — composite of dissolution, uniformity, hardness, friability, disintegration
- **Yield Score** — tablet weight consistency vs target
- **Performance Score** — moisture content deviation from optimal
- **Energy Efficiency** — inverted normalized energy consumption
- **Carbon Efficiency** — inverted normalized CO₂ emissions

### Pareto Front Identification
A batch is Pareto-optimal if no other batch dominates it across
all five objectives simultaneously. 21 of 60 batches (35%)
were identified as Pareto-optimal.

### Golden Signature Update Rule
A new batch triggers a GS update if:
1. Its composite score exceeds the current GS by ≥ 1% (configurable)
2. The human operator ACCEPTs the update via HITL workflow

### Energy Estimation
T001 time-series data used as calibration reference:
```
Est_Energy = f(Granulation_Time, Drying_Temp × Drying_Time,
               Compression_Force × Machine_Speed)
```
India grid carbon factor: 0.000233 kg CO₂ / Wh

---

## 🤝 Human-in-the-Loop Design

The HITL system satisfies all three problem statement requirements:

**1. Accept proposed changes**
Operator reviews parameter gap analysis and expected impact,
then clicks ACCEPT to log approval and trigger GS update evaluation.

**2. Reprioritize targets**
Operator adjusts objective weights in real-time via sliders.
New weights are logged and applied to future optimization runs.

**3. Maintain & reuse HITL inputs**
Every decision is persisted to `hitl_decisions.json` with
timestamp, batch ID, scenario, decision, reason, and weights.
This log is loaded on every subsequent recommendation call.

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | System status |
| POST | `/optimize` | Run optimization for scenario |
| GET | `/optimize/all` | All 3 scenarios |
| GET | `/golden-signature` | All GS summary |
| GET | `/golden-signature/{key}` | Specific GS |
| POST | `/hitl-decision` | Submit HITL decision |
| GET | `/hitl-history` | Full decision log |
| POST | `/recommend` | Get recommendations |
| GET | `/adaptive-targets` | Current targets |
| GET | `/history` | All batch history |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| pandas | 2.1.4 | Data processing |
| numpy | 1.26.4 | Numerical computing |
| scikit-learn | 1.3.2 | Normalization utilities |
| xgboost | 2.0.3 | ML framework |
| plotly | 5.18.0 | Interactive visualizations |
| streamlit | 1.31.0 | Dashboard UI |
| fastapi | 0.109.2 | REST API framework |
| uvicorn | 0.27.1 | ASGI server |
| scipy | 1.11.4 | Scientific computing |
| openpyxl | 3.1.2 | Excel file reading |

---

## 👥 Team

**Team VORTEX**
National AI & ML Hackathon — YUVAAN
Organized by Tinkerers' Lab, IIT Hyderabad
Powered by AVEVA

---

*Built with precision. Optimized for impact.*