"""
models/xgboost_model.py
-----------------------
XGBoost wrapper: build and train the final model with best Optuna params.

HOW XGBOOST WORKS (quick recap)
--------------------------------
XGBoost builds trees LEVEL-WISE — it splits all nodes at the same depth
before going deeper. This is safer and less prone to overfitting,
but slower than LightGBM's leaf-wise approach.

For multi-class, we use objective='multi:softprob' which outputs
a probability per class and picks the highest.
"""

import xgboost as xgb
from src.utils.logger import logger


def build_xgb_model(params: dict, n_classes: int) -> xgb.XGBClassifier:
    """Instantiate XGBClassifier with the given Optuna-tuned params."""
    model = xgb.XGBClassifier(
        **params,
        objective         = "multi:softprob",
        num_class         = n_classes,
        eval_metric       = "mlogloss",
        use_label_encoder = False,
        random_state      = 42,
        n_jobs            = -1,
    )
    logger.info(f"XGBoost model built with params: {params}")
    return model


def train_xgb(model: xgb.XGBClassifier, X_train, y_train) -> xgb.XGBClassifier:
    """Fit the model on training data."""
    logger.info("Training XGBoost ...")
    model.fit(X_train, y_train)
    logger.info("XGBoost training complete ✅")
    return model
