"""
utils/mlflow_utils.py
---------------------
MLflow helpers: experiment setup, metric/param/artifact logging.
"""

import os
import mlflow
from src.utils.logger import logger


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Point MLflow at the local (or remote) tracking store."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking URI : {tracking_uri}")
    logger.info(f"MLflow experiment   : {experiment_name}")


def log_params(params: dict) -> None:
    """Log a dictionary of hyperparameters to the active MLflow run."""
    mlflow.log_params(params)
    logger.info(f"Logged params: {params}")


def log_metrics(metrics: dict) -> None:
    """Log a dictionary of metrics to the active MLflow run."""
    mlflow.log_metrics(metrics)
    logger.info(f"Logged metrics: { {k: round(v, 4) for k, v in metrics.items()} }")


def log_artifact(path: str, remove_after: bool = True) -> None:
    """
    Log a local file as an MLflow artifact.
    Optionally deletes the local copy afterwards to keep the workspace clean.
    """
    mlflow.log_artifact(path)
    logger.info(f"Logged artifact: {path}")
    if remove_after and os.path.exists(path):
        os.remove(path)


def log_model_xgb(model, artifact_path: str = "model") -> None:
    import mlflow.xgboost
    mlflow.xgboost.log_model(model, artifact_path=artifact_path)
    logger.info("XGBoost model logged to MLflow")


def log_model_lgb(model, artifact_path: str = "model") -> None:
    import mlflow.lightgbm
    mlflow.lightgbm.log_model(model, artifact_path=artifact_path)
    logger.info("LightGBM model logged to MLflow")
