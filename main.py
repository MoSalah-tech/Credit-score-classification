"""
main.py
-------
Single entry point for the Credit Score Classification pipeline.

Usage
-----
  python main.py                          # uses defaults from config.yaml
  python main.py --trials 50             # more Optuna trials
  python main.py --data data/train.csv   # explicit data path

After running:
  mlflow ui --backend-store-uri mlflow_runs
  → open http://127.0.0.1:5000
"""

import argparse
import yaml
import mlflow

from src.utils.logger import logger
from src.utils.mlflow_utils import (
    setup_mlflow, log_params, log_metrics, log_artifact,
    log_model_xgb, log_model_lgb,
)

from src.data.loader   import load_train, show_basic_info
from src.data.cleaner  import clean

from src.features.encoder  import encode
from src.features.splitter import split

from src.models.tuner          import xgb_objective, lgb_objective, run_study
from src.models.xgboost_model  import build_xgb_model, train_xgb
from src.models.lightgbm_model import build_lgb_model, train_lgb

from src.evaluation.metrics    import compute_metrics
from src.evaluation.visualizer import plot_confusion_matrix


# =============================================================================
# Load config
# =============================================================================

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Single model run (train → evaluate → log to MLflow)
# =============================================================================

def run_model(
    model_name: str,
    model,
    best_params: dict,
    X_train, X_test,
    y_train, y_test,
    class_names: list,
    cfg: dict,
) -> dict:
    """Train one model, log everything to MLflow, return metrics dict."""

    setup_mlflow(
        tracking_uri    = cfg["mlflow"]["tracking_uri"],
        experiment_name = cfg["mlflow"]["experiment_name"],
    )

    with mlflow.start_run(run_name=model_name):
        # Train
        if "XGBoost" in model_name:
            model = train_xgb(model, X_train, y_train)
        else:
            model = train_lgb(model, X_train, y_train)

        y_pred = model.predict(X_test)

        # Metrics
        metrics = compute_metrics(y_pred=y_pred, y_true=y_test, class_names=class_names)

        # MLflow — log params + metrics
        log_params(best_params)
        log_metrics(metrics)

        # MLflow — confusion matrix artifact
        cm_path = plot_confusion_matrix(
            y_true      = y_test,
            y_pred      = y_pred,
            class_names = class_names,
            model_name  = model_name,
            save_dir    = cfg["outputs"]["plots_dir"],
        )
        log_artifact(cm_path, remove_after=False)   # keep local copy too

        # MLflow — model artifact
        if "XGBoost" in model_name:
            log_model_xgb(model)
        else:
            log_model_lgb(model)

    return metrics


# =============================================================================
# Main Pipeline
# =============================================================================

def main(data_path: str = None, n_trials: int = None):
    cfg = load_config()

    # Allow CLI overrides
    data_path = data_path or cfg["data"]["train_path"]
    n_trials  = n_trials  or cfg["training"]["n_trials"]

    logger.info("=" * 60)
    logger.info("  Credit Score Classification Pipeline")
    logger.info("=" * 60)

    # 1. Load
    df = load_train(data_path)
    show_basic_info(df)

    # 2. Clean
    df = clean(
        df,
        drop_columns         = cfg["data"]["drop_columns"],
        numeric_fill_strategy= cfg["features"]["numeric_fill_strategy"],
        target_col           = cfg["features"]["target_column"],
    )

    # 3. Encode
    X, y, le_target = encode(df, target_col=cfg["features"]["target_column"])
    class_names = list(le_target.classes_)
    n_classes   = len(class_names)

    # 4. Split
    X_train, X_test, y_train, y_test = split(
        X, y,
        test_size    = cfg["data"]["test_size"],
        random_state = cfg["data"]["random_state"],
    )

    all_results = {}

    # -------------------------------------------------------------------------
    # 5a. XGBoost — Optuna tuning → train → MLflow
    # -------------------------------------------------------------------------
    logger.info("\n🔍 Tuning XGBoost with Optuna ...")
    best_xgb_params = run_study(
        objective_fn = lambda t: xgb_objective(t, X_train, y_train, n_classes),
        study_name   = "xgboost_credit_score",
        n_trials     = n_trials,
    )
    xgb_model = build_xgb_model(best_xgb_params, n_classes)
    all_results["XGBoost"] = run_model(
        model_name  = "XGBoost",
        model       = xgb_model,
        best_params = best_xgb_params,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        class_names = class_names,
        cfg         = cfg,
    )

    # -------------------------------------------------------------------------
    # 5b. LightGBM — Optuna tuning → train → MLflow
    # -------------------------------------------------------------------------
    logger.info("\n🔍 Tuning LightGBM with Optuna ...")
    best_lgb_params = run_study(
        objective_fn = lambda t: lgb_objective(t, X_train, y_train),
        study_name   = "lightgbm_credit_score",
        n_trials     = n_trials,
    )
    lgb_model = build_lgb_model(best_lgb_params)
    all_results["LightGBM"] = run_model(
        model_name  = "LightGBM",
        model       = lgb_model,
        best_params = best_lgb_params,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        class_names = class_names,
        cfg         = cfg,
    )

    # -------------------------------------------------------------------------
    # 6. Final comparison
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("  📊 FINAL MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info(f"  {'Model':<15} {'Accuracy':>10} {'F1 Weighted':>13}")
    logger.info(f"  {'-'*40}")
    for name, res in all_results.items():
        logger.info(f"  {name:<15} {res['accuracy']:>10.4f} {res['f1_weighted']:>13.4f}")

    winner = max(all_results, key=lambda k: all_results[k]["f1_weighted"])
    logger.info(f"\n  🏆 Winner by Weighted F1: {winner}")
    logger.info("=" * 60)
    logger.info("\n✅ Done! Launch MLflow UI with:")
    logger.info(f"   mlflow ui --backend-store-uri {cfg['mlflow']['tracking_uri']}")
    logger.info("   Then open http://127.0.0.1:5000")


# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Score ML Pipeline")
    parser.add_argument("--data",   type=str, default=None,
                        help="Path to train.csv (overrides config.yaml)")
    parser.add_argument("--trials", type=int, default=None,
                        help="Optuna trials per model (overrides config.yaml)")
    args = parser.parse_args()
    main(data_path=args.data, n_trials=args.trials)
