"""
evaluation/visualizer.py
------------------------
Generate plots and save them to outputs/plots/.
Currently: Confusion Matrix heatmap.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.utils.logger import logger


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list,
    model_name: str,
    save_dir: str = "outputs/plots",
) -> str:
    """
    Draw and save a confusion matrix heatmap.

    Returns the file path so it can be logged as an MLflow artifact.
    """
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    filename = f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)

    logger.info(f"Confusion matrix saved → {path}")
    return path
