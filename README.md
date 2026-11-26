# MP291 - Parcel Delivery Days (Dehradun)

**Course**: SCSE3040 MLOps Mini Project  
**Group ID**: G104  
**Problem ID**: MP291  
**Batch**: 9  
**Faculty**: Dr. Pulkit Dwivedi  

---

## 👥 Group Members

| Enrolment No. | Name | 
| :--- | :--- | 
| **S24CSEU0875** | Chinta Sri CharanReddy | 
| **S24CSEU0951** | AadrikaRakheja |
| **S24CSEU0957** | GaytriBembadeSantosh |
| **S24CSEU0942** | suryasinghpratap |

---

## 📦 Project Stages Overview

This repository covers **Stage 1**, **Stage 2**, and **Stage 3** of the MLOps pipeline:

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

- **Stage 3 (Tue 24 Nov - Thu 26 Nov)**:
  - Production FastAPI service (`app/main.py`) exposing `GET /health` and `POST /predict`
  - Client validation rules in Pydantic schema (`declared_value_rs <= 35000`, `hub_backlog >= 0`)
  - Structured JSON logging to stdout (one JSON line per prediction, zero PII)
  - Containerization with Docker (`Dockerfile`, `.dockerignore`, image `mp291-api:1.0`)
  - Automated CI workflow on GitHub Actions (`.github/workflows/ci.yml`) running pytest
  - Kubernetes manifests (`k8s/deployment.yaml`, `k8s/service.yaml`) with 3 replicas and health probes
  - API endpoint unit tests using `TestClient` (`tests/test_api.py`)

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

## 🚀 Stage 3: Prediction Service, CI/CD & Deployment

1. **Prediction API (`app/main.py`)**:
   - Built with **FastAPI** to serve real-time predictions for Dehradun Speed Parcel Hub.
   - **`GET /health`**: Health check probe returning `{"status": "ok"}`.
   - **`POST /predict`**: Accepts exactly the 8 required fields:
     `distance_km`, `weight_kg`, `hub_backlog`, `pin_code_tier`, `declared_value_rs`, `transfers`, `service`, `destination`.
   - **Client Validation Rules (Pydantic `Field`)**:
     1. `declared_value_rs`: Above 35,000 is impossible for the client. Rejected with status code **422** (`Field(..., le=35000)`).
     2. `hub_backlog`: Below 0 is a typing mistake. Rejected with status code **422** (`Field(..., ge=0)`).
   - **JSON Logging**: Implements a custom `JSONFormatter` writing one single-line JSON log per inference request directly to `stdout` without logging customer PII.

2. **Containerization (`Dockerfile`, `.dockerignore`)**:
   - Built on `python:3.12-slim`.
   - Optimized layer caching: installs `requirements.txt` before copying code.
   - Exposes container port `8000` and starts uvicorn server.
   - Clean `.dockerignore` excludes git, dvc, raw data, test caches, and temp artifacts.
   - Image target: `mp291-api:1.0`.

3. **Continuous Integration (`.github/workflows/ci.yml`)**:
   - Triggers on every `push` and `pull_request` (starts with literal `on:`).
   - Sets up Python 3.12, installs pinned `requirements.txt`, and runs full `pytest -v` suite.
   - Fully independent of raw CSV files (test fixtures synthesize in-memory inputs).

4. **Kubernetes Configuration (`k8s/`)**:
   - **Deployment (`k8s/deployment.yaml`)**:
     - Name: `mp291-api`
     - Replicas: `3`
     - Container port: `8000`
     - Probes: Readiness probe (`initialDelaySeconds: 5`) and Liveness probe (`initialDelaySeconds: 15`, waiting longer) on `/health`.
     - Matching selector labels: `app: mp291-api`.
   - **Service (`k8s/service.yaml`)**:
     - Name: `mp291-svc`
     - Type: `ClusterIP`
     - Port forwarding: Port `80` routing to target port `8000`.

---

## 📂 Repository Layout

```
.
├── .dockerignore                # Docker ignore rules
├── .dvc/
│   ├── config                   # DVC configuration from dvc init
│   └── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow (pytest on push)
├── app/
│   ├── __init__.py
│   └── main.py                  # FastAPI service: POST /predict & GET /health
├── data/
│   └── raw/
│       ├── .gitignore           # Ignores raw train.csv and later.csv from git
│       ├── train.csv.dvc        # DVC tracker for training data
│       ├── later.csv.dvc        # DVC tracker for later data
│       └── holdout_features.csv # Holdout features (tracked in git)
├── Dockerfile                   # python:3.12-slim container definition
├── k8s/
│   ├── deployment.yaml          # Kubernetes Deployment (3 replicas, probes)
│   └── service.yaml             # Kubernetes Service (port 80 -> 8000)
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
│   ├── test_api.py              # TestClient tests for FastAPI endpoints & rules
│   ├── test_cleaning.py         # Unit tests for data cleaning logic
│   ├── test_data_checks.py      # Unit tests for Stage 1 data checks
│   └── test_pipeline.py         # Unit tests for scikit-learn pipeline
├── findings.yaml                # Findings form at root
├── GROUP.md                     # Group details
├── mlflow.db                    # SQLite MLflow database (committed to git)
├── params.yaml                  # Settings read with yaml.safe_load
├── PROBLEM.md                   # Problem description
├── pytest.ini                   # Pytest path settings
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

### 3. Run Automated Tests
```bash
pytest -v
```

### 4. Run the Prediction API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Test `/health`:
```bash
curl -X GET http://localhost:8000/health
```

Test `/predict`:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "distance_km": 350.5,
    "weight_kg": 3.2,
    "hub_backlog": 15,
    "pin_code_tier": 2,
    "declared_value_rs": 4500.0,
    "transfers": 1,
    "service": "standard",
    "destination": "Tier-2"
  }'
```

### 5. Docker Build and Run
```bash
# Build the container image
docker build -t mp291-api:1.0 .

# Run the container
docker run -d -p 8000:8000 --name mp291-container mp291-api:1.0
```
