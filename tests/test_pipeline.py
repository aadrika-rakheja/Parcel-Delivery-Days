"""
Tests for pipeline construction and model inference on synthetic in-memory data.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from mp291.train import build_pipeline


def test_build_pipeline_and_predict():
    num_cols = ["distance_km", "weight_kg", "hub_backlog", "pin_code_tier", "declared_value_rs", "transfers"]
    cat_cols = ["service", "destination"]

    # Synthetic training data
    df = pd.DataFrame({
        "distance_km": [100.0, 500.0, 1000.0, 200.0, 800.0],
        "weight_kg": [1.5, 3.0, 10.0, 0.5, 5.0],
        "hub_backlog": [10, 50, 100, 20, 80],
        "pin_code_tier": [1, 2, 3, 1, 2],
        "declared_value_rs": [500.0, 2000.0, 10000.0, 300.0, 4500.0],
        "transfers": [0, 1, 3, np.nan, 2],
        "service": ["economy", "standard", "express", "economy", "standard"],
        "destination": ["Metro", "Tier-2", "Rural", "Metro", "Tier-2"],
    })
    y = np.array([2.0, 3.5, 5.0, 2.2, 4.1])

    pipe = build_pipeline(Ridge(alpha=1.0), num_cols, cat_cols)
    pipe.fit(df, y)

    # Predict on new synthetic row with missing transfer and unseen formatting
    test_sample = pd.DataFrame({
        "distance_km": [250.0],
        "weight_kg": [2.0],
        "hub_backlog": [30],
        "pin_code_tier": [2],
        "declared_value_rs": [1500.0],
        "transfers": [np.nan],
        "service": ["express"],
        "destination": ["Metro"],
    })

    pred = pipe.predict(test_sample)
    assert len(pred) == 1
    assert pred[0] > 0
