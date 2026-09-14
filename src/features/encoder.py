"""
features/encoder.py
-------------------
Label-encodes categorical columns and the target variable.
Returns X, y, and the target LabelEncoder (needed to map predictions back).
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import logger


def encode(
    df: pd.DataFrame,
    target_col: str = "Credit_Score",
):
    """
    1. Label-encode the target column.
    2. Label-encode any remaining object/categorical columns.

    Returns
    -------
    X           : pd.DataFrame  — feature matrix
    y           : pd.Series     — encoded target
    le_target   : LabelEncoder  — fitted on the target (use .classes_ to see labels)
    """
    df = df.copy()

    # --- Target ---
    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col])
    logger.info(f"Target classes: {list(le_target.classes_)}")

    y = df.pop(target_col)
    X = df

    # --- Remaining categoricals ---
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))
        logger.info(f"  Label-encoded: {col}")

    logger.info(f"Encoding complete → {X.shape[1]} features, {y.nunique()} classes")
    return X, y, le_target
