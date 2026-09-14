"""Tests for feature encoding."""
import pandas as pd
from src.features.encoder import encode


def test_encode_target_classes():
    df = pd.DataFrame({
        "Feature1": [1, 2, 3],
        "Occupation": ["Engineer", "Doctor", "Lawyer"],
        "Credit_Score": ["Good", "Poor", "Standard"],
    })
    X, y, le = encode(df, target_col="Credit_Score")
    assert set(le.classes_) == {"Good", "Poor", "Standard"}
    assert "Credit_Score" not in X.columns
    assert len(y) == 3


def test_encode_no_object_columns_remaining():
    df = pd.DataFrame({
        "Cat": ["A", "B", "A"],
        "Num": [1, 2, 3],
        "Credit_Score": ["Good", "Poor", "Standard"],
    })
    X, y, le = encode(df)
    assert X.select_dtypes(include="object").empty
