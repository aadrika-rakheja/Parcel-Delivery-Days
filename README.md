# MP291 - Parcel Delivery Days (Dehradun)

**Course**: SCSE3040 MLOps Mini Project  
**Group ID**: G104  
**Problem ID**: MP291  
**Batch**: 9  
**Faculty**: Dr. Pulkit Dwivedi  

---

## 👥 Group Members

| Enrolment No. | Name | Role |
| :--- | :--- | :--- |
| **S24CSEU0875** | Chinta Sri CharanReddy | Data Lead |
| **S24CSEU0951** | AadrikaRakheja | Model Lead |
| **S24CSEU0957** | GaytriBembadeSantosh | Service Lead |
| **S24CSEU0942** | suryasinghpratap | Ops Lead |

---

## 📦 Project Stages Overview

This repository currently covers **Stage 1** and **Stage 2** of the MLOps pipeline:

- **Stage 1 (Wed 18 Nov - Fri 20 Nov)**:
  - GitHub repository initialization and structure
  - DVC data versioning (`.dvc/`, `data/raw/train.csv.dvc`, `data/raw/later.csv.dvc`)
  - Systematic investigation and reporting of the seven data problems in `data/raw/train.csv` (`src/mp291/data_checks.py`, `reports/findings.yaml`)

- **Stage 2 (Sat 21 Nov - Mon 23 Nov)**:
  - Data cleaning pipeline (`src/mp291/clean.py`)
  - Model training pipeline with MLflow experiment tracking (`src/mp291/train.py`)
  - Experiment logging with SHA-256 data fingerprint (`data_sha256`) and MAE metric tracking to SQLite (`mlflow.db`)
  - Model registry: Best model registered under `mp291_delivery_predictor` with alias `@champion`
  - Exported model artifact (`models/model.joblib`)
  - Automated unit and pipeline tests (`tests/`)

---

## 🔍 Stage 1: Data Problem Diagnostics (`train.csv`)

Our diagnostic code in `src/mp291/data_checks.py` audited all 7 potential data quality problems:

| Data Issue | Present? | Findings & Evidence |
| :--- | :---: | :--- |
| **Leaking Column** | **True** | `days_from_tracking_page` is absent from `holdout_features.csv` and has a correlation of 0.9987 with `delivery_days`. Dropped from features. |
| **Unit Mixup** | **False** | Verified distributions across all destinations (`Metro`, `Tier-2`, `Rural`). No 10x/100x unit shifts found. |
| **Sentinel Value** | **True** | Column `transfers` has sentinel `-999` across 33 rows. Cleaned and imputed with median. |
| **Duplicate Rows** | **True** | Exactly 30 duplicate rows identified and removed (`df.drop_duplicates()`). |
| **Stuck Sensor** | **False** | No continuous feature has identical consecutive readings across dates. |
| **Target Typos** | **True** | Exactly 12 rows had `delivery_days > 20` due to a 10x extra-zero typo. Fixed by dividing by 10. |
| **Spelling Variants** | **True** | Column `service` contained dirty variants (`eco.`, `sta.`, `exp.`, uppercase). Cleaned to 3 true levels: `economy`, `standard`, `express`. |

Findings are documented in `reports/findings.yaml` and `findings.yaml`.

---

## 🏗️ Stage 2: Cleaning, MLflow Tracking & Model Registry

1. **Cleaning Logic (`src/mp291/clean.py`)**:
   - Removes duplicate rows
   - Fixes the 12 target typos by dividing by 10
   - Drops `days_from_tracking_page`
   - Replaces `-999` in `transfers` with `NaN` (imputed with median in pipeline)
   - Standardizes `service` categories to 3 clean levels

2. **MLflow Tracking & Registry (`src/mp291/train.py`)**:
   - Tracking backend: `sqlite:///mlflow.db`
   - Data fingerprint logged: `data_sha256: bbac2b611822071d0fc22f32afd90a5527fcf153d7ddc91909bc5947af6f0383`
   - Candidate models evaluated across 4 runs:
     - `Ridge (alpha=10.0)` -> Val MAE: **0.2610** (Best)
     - `Ridge (alpha=1.0)` -> Val MAE: **0.2618**
     - `GradientBoosting` -> Val MAE: **0.3231**
     - `RandomForest` -> Val MAE: **0.3720**
   - Registered model: `mp291_delivery_predictor` (Version 1)
   - Alias: `@champion`
   - Serialized model saved to: `models/model.joblib`

---

## 📂 Repository Layout

```
.
├── .dvc/
│   ├── config                   # DVC configuration from dvc init
│   └── .gitignore
├── data/
│   └── raw/
│       ├── .gitignore           # Ignores raw train.csv and later.csv from git
│       ├── train.csv.dvc        # DVC tracker for training data
│       ├── later.csv.dvc        # DVC tracker for later data
│       └── holdout_features.csv # Holdout features (tracked in git)
├── models/
│   └── model.joblib             # Serialized champion model
├── reports/
│   └── findings.yaml            # Form filled for Stage 1 & Stage 2
├── src/
│   └── mp291/
│       ├── __init__.py
│       ├── clean.py             # Data cleaning pipeline
│       ├── data_checks.py       # Code checking the 7 data issues
│       └── train.py             # Training & MLflow tracking script
├── tests/
│   ├── test_cleaning.py         # Unit tests for data cleaning logic
│   ├── test_data_checks.py      # Unit tests for Stage 1 data checks
│   └── test_pipeline.py         # Unit tests for scikit-learn pipeline
├── findings.yaml                # Findings form at root
├── GROUP.md                     # Group details
├── mlflow.db                    # SQLite MLflow database (committed to git)
├── params.yaml                  # Settings read with yaml.safe_load
├── PROBLEM.md                   # Problem description
├── README.md                    # Group id, problem id, member details
├── requirements.txt             # Pinned project dependencies
└── STUDENT_HANDBOOK.md          # Assignment rules and guidelines
```

---

## 🧪 Running Locally

### 1. Run Data Checks
```bash
python src/mp291/data_checks.py
```

### 2. Train Models and Log to MLflow
```bash
python src/mp291/train.py
```

### 3. Run Pytest Suite
```bash
pytest
```
