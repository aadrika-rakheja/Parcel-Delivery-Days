"""
Unit tests for data cleaning logic without requiring external CSV files.
"""

import numpy as np
import pandas as pd
from mp291.clean import clean_dataframe


def test_clean_dataframe_removes_duplicates():
    df = pd.DataFrame({
        "record_id": ["R1", "R1"],
        "date": ["2026-01-01", "2026-01-01"],
        "distance_km": [100, 100],
        "weight_kg": [2.0, 2.0],
        "hub_backlog": [50, 50],
        "pin_code_tier": [1, 1],
        "declared_value_rs": [1000, 1000],
        "transfers": [1, 1],
        "service": ["standard", "standard"],
        "destination": ["Metro", "Metro"],
        "delivery_days": [3.0, 3.0],
    })
    cleaned = clean_dataframe(df, is_train=True)
    assert len(cleaned) == 1


def test_clean_dataframe_fixes_target_typos():
    df = pd.DataFrame({
        "record_id": ["R1", "R2"],
        "delivery_days": [45.0, 4.5],
        "transfers": [1, 2],
        "service": ["express", "economy"],
    })
    cleaned = clean_dataframe(df, is_train=True)
    # 45.0 should become 4.5
    assert cleaned.loc[0, "delivery_days"] == 4.5
    assert cleaned.loc[1, "delivery_days"] == 4.5


def test_clean_dataframe_handles_sentinel_and_spelling():
    df = pd.DataFrame({
        "transfers": [-999, 2],
        "service": [" eco. ", "sta."],
        "days_from_tracking_page": [5.2, 3.1],
    })
    cleaned = clean_dataframe(df, is_train=False)
    assert pd.isna(cleaned.loc[0, "transfers"])
    assert cleaned.loc[0, "service"] == "economy"
    assert cleaned.loc[1, "service"] == "standard"
    assert "days_from_tracking_page" not in cleaned.columns
