"""
Data cleaning and preprocessing module for MP291.
Cleans raw parcel delivery data based on verified data issues.
"""

import numpy as np
import pandas as pd


SERVICE_MAPPING = {
    "eco.": "economy",
    "sta.": "standard",
    "exp.": "express",
}


def clean_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Clean the input DataFrame by fixing duplicates, target typos,
    sentinel values, spelling variants, and dropping leaking columns.
    """
    cleaned = df.copy()

    # 1. Deduplication (for training data)
    if is_train:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 2. Fix target typos (values typed 10 times too big)
    if is_train and "delivery_days" in cleaned.columns:
        mask = cleaned["delivery_days"] > 20
        cleaned.loc[mask, "delivery_days"] = cleaned.loc[mask, "delivery_days"] / 10.0

    # 3. Drop leaking column
    if "days_from_tracking_page" in cleaned.columns:
        cleaned = cleaned.drop(columns=["days_from_tracking_page"])

    # 4. Handle sentinel value in transfers
    if "transfers" in cleaned.columns:
        cleaned["transfers"] = cleaned["transfers"].replace(-999, np.nan)

    # 5. Clean spelling variants in service
    if "service" in cleaned.columns:
        cleaned["service"] = cleaned["service"].astype(str).str.strip().str.lower()
        cleaned["service"] = cleaned["service"].replace(SERVICE_MAPPING)

    # 6. Standardize destination strings
    if "destination" in cleaned.columns:
        cleaned["destination"] = cleaned["destination"].astype(str).str.strip()

    return cleaned


def load_and_clean_data(csv_path: str, is_train: bool = True) -> pd.DataFrame:
    """Load CSV from path and apply clean_dataframe."""
    df = pd.read_csv(csv_path)
    return clean_dataframe(df, is_train=is_train)
