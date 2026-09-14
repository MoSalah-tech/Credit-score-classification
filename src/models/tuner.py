"""
models/tuner.py
---------------
Optuna objective functions for XGBoost and LightGBM.
Each objective runs 3-fold stratified CV and returns weighted F1.
"""

import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb
from src.utils.logger import logger

optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# XGBoost Objective
# =============================================================================

def xgb_objective(trial, X_train, y_train, n_classes: int) -> float:
    """
    Hyperparameter search space for XGBoost.

    Key params explained:
    - max_depth       : how deep each tree grows (deeper = more complex)
    - learning_rate   : shrinkage — lower = slower but more accurate
    - subsample       : fraction of rows per tree (prevents overfit)
    - colsample_bytree: fraction of features per tree
    - reg_alpha/lambda: L1 and L2 regularization
    """
    params = {
        "n_estimators"      : trial.suggest_int("n_estimators", 100, 500),
        "max_depth"         : trial.suggest_int("max_depth", 3, 10),
        "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample"         : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha"         : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"        : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "objective"         : "multi:softprob",
        "num_class"         : n_classes,
        "eval_metric"       : "mlogloss",
        "use_label_encoder" : False,
        "random_state"      : 42,
        "n_jobs"            : -1,
    }
    model = xgb.XGBClassifier(**params)
    cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train,
                            cv=cv, scoring="f1_weighted", n_jobs=-1).mean()
    return score


# =============================================================================
# LightGBM Objective
# =============================================================================

def lgb_objective(trial, X_train, y_train) -> float:
    """
    Hyperparameter search space for LightGBM.

    Extra LightGBM-specific params vs XGBoost:
    - num_leaves        : max leaves per tree (key param — higher = more complex)
    - min_child_samples : min data in a leaf (prevents overfit on small leaves)

    These exist because LightGBM grows leaf-wise, not level-wise like XGBoost,
    so you control complexity through leaves rather than depth alone.
    """
    params = {
        "n_estimators"      : trial.suggest_int("n_estimators", 100, 500),
        "max_depth"         : trial.suggest_int("max_depth", 3, 10),
        "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves"        : trial.suggest_int("num_leaves", 20, 150),
        "subsample"         : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha"         : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"        : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_samples" : trial.suggest_int("min_child_samples", 5, 100),
        "objective"         : "multiclass",
        "random_state"      : 42,
        "n_jobs"            : -1,
        "verbose"           : -1,
    }
    model = lgb.LGBMClassifier(**params)
    cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train,
                            cv=cv, scoring="f1_weighted", n_jobs=-1).mean()
    return score


# =============================================================================
# Run a Study
# =============================================================================

def run_study(objective_fn, study_name: str, n_trials: int) -> dict:
    """Create and run an Optuna study, return the best params dict."""
    from optuna.samplers import TPESampler
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name=study_name,
    )
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"[{study_name}] Best F1: {study.best_value:.4f}")
    logger.info(f"[{study_name}] Best params: {study.best_params}")
    return study.best_params
