"""
Tests for Stage 1 data checks logic using synthetic data.
"""

import numpy as np
import pandas as pd
from mp291.clean import clean_dataframe


def test_target_typos_detection_and_fix():
    # Synthetic targets with typos > 20
    df = pd.DataFrame({
        "record_id": ["R1", "R2", "R3", "R4", "R5"],
        "delivery_days": [3.5, 4.2, 55.0, 78.0, 2.1],
        "transfers": [1, 2, 0, 1, 1],
        "service": ["economy", "standard", "express", "economy", "standard"],
        "destination": ["Metro", "Tier-2", "Rural", "Metro", "Tier-2"]
    })
    typos_mask = df["delivery_days"] > 20
    assert typos_mask.sum() == 2

    cleaned = clean_dataframe(df, is_train=True)
    assert (cleaned["delivery_days"] > 20).sum() == 0
    assert cleaned.loc[2, "delivery_days"] == 5.5
    assert cleaned.loc[3, "delivery_days"] == 7.8


def test_sentinel_replacement():
    df = pd.DataFrame({
        "record_id": ["R1", "R2", "R3", "R4", "R5"],
        "transfers": [-999, 0, 1, -999, 3],
        "service": ["economy", "standard", "express", "standard", "express"],
    })
    cleaned = clean_dataframe(df, is_train=True)
    assert cleaned["transfers"].isna().sum() == 2
