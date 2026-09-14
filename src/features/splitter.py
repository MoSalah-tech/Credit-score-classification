"""
features/splitter.py
--------------------
Stratified train/test split — keeps class proportions equal in both sets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import logger


def split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Stratified split so all three Credit Score classes
    (Good / Standard / Poor) are proportionally represented.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    logger.info(f"Train size : {X_train.shape[0]:,}")
    logger.info(f"Test size  : {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test
