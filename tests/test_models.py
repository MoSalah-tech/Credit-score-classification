"""Smoke tests — make sure models train and predict without errors."""
import numpy as np
from src.models.xgboost_model  import build_xgb_model, train_xgb
from src.models.lightgbm_model import build_lgb_model, train_lgb

X = np.random.rand(100, 10)
y = np.random.randint(0, 3, 100)


def test_xgb_trains_and_predicts():
    params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
              "subsample": 0.8, "colsample_bytree": 0.8,
              "reg_alpha": 0.1, "reg_lambda": 1.0}
    model = build_xgb_model(params, n_classes=3)
    model = train_xgb(model, X, y)
    preds = model.predict(X)
    assert len(preds) == 100
    assert set(preds).issubset({0, 1, 2})


def test_lgb_trains_and_predicts():
    params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
              "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8,
              "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_samples": 5}
    model = build_lgb_model(params)
    model = train_lgb(model, X, y)
    preds = model.predict(X)
    assert len(preds) == 100
    assert set(preds).issubset({0, 1, 2})
