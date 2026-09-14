"""
models/lightgbm_model.py
------------------------
LightGBM wrapper: build and train the final model with best Optuna params.

HOW LIGHTGBM WORKS (quick recap)
----------------------------------
LightGBM builds trees LEAF-WISE — at each step it picks the single leaf
with the highest loss reduction and splits only that one.
This makes it faster and often more accurate, but riskier on small datasets
since it can grow very unbalanced, deep trees.

Key extra param vs XGBoost:
  num_leaves — controls max complexity (replaces max_depth as primary lever).
  Always set num_leaves < 2^max_depth to keep trees balanced.
"""

import lightgbm as lgb
from src.utils.logger import logger


def build_lgb_model(params: dict) -> lgb.LGBMClassifier:
    """Instantiate LGBMClassifier with the given Optuna-tuned params."""
    model = lgb.LGBMClassifier(
        **params,
        objective    = "multiclass",
        random_state = 42,
        n_jobs       = -1,
        verbose      = -1,
    )
    logger.info(f"LightGBM model built with params: {params}")
    return model


def train_lgb(model: lgb.LGBMClassifier, X_train, y_train) -> lgb.LGBMClassifier:
    """Fit the model on training data."""
    logger.info("Training LightGBM ...")
    model.fit(X_train, y_train)
    logger.info("LightGBM training complete ✅")
    return model
