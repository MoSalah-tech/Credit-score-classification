"""Tests for data cleaning functions."""
import pandas as pd
import numpy as np
import pytest
from src.data.cleaner import drop_irrelevant_columns, fix_age, impute_missing


def test_drop_irrelevant_columns():
    df = pd.DataFrame({"ID": [1], "Name": ["Alice"], "Age": [25]})
    result = drop_irrelevant_columns(df, ["ID", "Name"])
    assert "ID" not in result.columns
    assert "Name" not in result.columns
    assert "Age" in result.columns


def test_fix_age_clamps_negatives():
    df = pd.DataFrame({"Age": [-500, 25, 150, 30]})
    result = fix_age(df)
    assert result["Age"].isna().sum() == 2   # -500 and 150 → NaN
    assert result["Age"].dropna().between(1, 100).all()


def test_impute_missing_numeric():
    df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0], "Credit_Score": ["Good"] * 4})
    result = impute_missing(df, numeric_strategy="median")
    assert result["A"].isna().sum() == 0
    assert result["A"].iloc[2] == 2.0   # median of [1,2,4] = 2
