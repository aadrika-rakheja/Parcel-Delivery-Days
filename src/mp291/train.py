"""
Model training and MLflow tracking module for MP291.
Trains candidate models on cleaned train.csv, logs runs to SQLite MLflow,
registers the champion model, and serializes to models/model.joblib.
"""

import hashlib
import os
import sys
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

from mp291.clean import load_and_clean_data


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_pipeline(regressor, num_cols, cat_cols):
    """Build full preprocessing and regression pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])


def train_v1(params_path="params.yaml"):
    """Train candidate models, log with MLflow, and register the champion."""
    with open(params_path, "r") as f:
        config = yaml.safe_load(f)

    raw_train_path = config["data"]["raw_train_path"]
    train_cfg = config["train"]
    model_cfg = config["model"]

    data_sha256 = compute_sha256(raw_train_path)
    print(f"Computed data_sha256: {data_sha256}")

    # Load and clean data
    df = load_and_clean_data(raw_train_path, is_train=True)
    num_cols = train_cfg["numeric_cols"]
    cat_cols = train_cfg["categorical_cols"]
    target_col = train_cfg["target_col"]

    X = df[num_cols + cat_cols]
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=train_cfg.get("test_size", 0.2),
        random_state=train_cfg.get("random_state", 42)
    )

    # MLflow Setup
    tracking_uri = model_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = model_cfg.get("experiment_name", "mp291_parcel_delivery")
    mlflow.set_experiment(experiment_name)

    client = MlflowClient()

    candidates = [
        ("ridge_alpha_1.0", Ridge(alpha=1.0, random_state=42), {"model_type": "Ridge", "alpha": 1.0}),
        ("gradient_boosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42), {"model_type": "GradientBoosting", "n_estimators": 100, "max_depth": 5}),
        ("random_forest", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42), {"model_type": "RandomForest", "n_estimators": 100, "max_depth": 8}),
        ("ridge_alpha_10.0", Ridge(alpha=10.0, random_state=42), {"model_type": "Ridge", "alpha": 10.0}),
    ]

    best_run_id = None
    best_mae = float("inf")
    best_pipeline = None

    for run_name, regressor, params in candidates:
        with mlflow.start_run(run_name=run_name) as run:
            pipe = build_pipeline(regressor, num_cols, cat_cols)
            pipe.fit(X_train, y_train)

            train_preds = pipe.predict(X_train)
            val_preds = pipe.predict(X_val)

            train_mae = float(mean_absolute_error(y_train, train_preds))
            val_mae = float(mean_absolute_error(y_val, val_preds))

            # Logging params and metrics
            mlflow.log_param("data_sha256", data_sha256)
            for k, v in params.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("val_mae", val_mae)
            mlflow.log_metric("mae", val_mae)  # ensures an 'mae' metric exists

            # Log model
            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                serialization_format="cloudpickle"
            )

            print(f"Run {run_name} ({run.info.run_id}) - Val MAE: {val_mae:.4f}")

            if val_mae < best_mae:
                best_mae = val_mae
                best_run_id = run.info.run_id
                best_pipeline = pipe

    print(f"\nBest Run ID: {best_run_id} with MAE: {best_mae:.4f}")

    # Register best model
    registered_model_name = model_cfg.get("registered_model_name", "mp291_delivery_predictor")
    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(model_uri, registered_model_name)
    v_num = int(model_version.version)

    # Set champion alias
    client.set_registered_model_alias(registered_model_name, "champion", v_num)
    print(f"Model version {v_num} registered and tagged with alias '@champion'")

    # Save to models/model.joblib
    os.makedirs("models", exist_ok=True)
    model_path = model_cfg.get("model_path", "models/model.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"Champion model saved to {model_path}")

    return {
        "experiment_name": experiment_name,
        "best_run_id": best_run_id,
        "registered_model_name": registered_model_name,
        "champion_version": v_num,
        "best_mae": best_mae,
    }


if __name__ == "__main__":
    train_v1()
