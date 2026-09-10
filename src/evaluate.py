import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for Colab, Kaggle, and headless servers
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from config import RESULTS_DIR

_DPI = 150  # Publication-quality figure resolution


def evaluate_model(y_true, y_pred, y_prob, model_name: str, dataset_name: str) -> dict:
    """
    Compute and persist all classification metrics.

    Metrics computed:
      - Accuracy, Precision, Recall, F1-Score  (macro + per-class via report)
      - ROC AUC
      - Confusion Matrix (printed + saved as PNG)

    Results are appended to a CSV file under RESULTS_DIR.

    Returns:
        dict with scalar metric values.
    """
    metrics = {
        "Dataset":    dataset_name,
        "Model":      model_name,
        "Accuracy":   round(accuracy_score(y_true, y_pred), 6),
        # average='weighted' matches the weighted avg row of classification_report,
        # which is the standard scalar summary for imbalanced datasets.
        "Precision":  round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 6),
        "Recall":     round(recall_score(y_true, y_pred,    average="weighted", zero_division=0), 6),
        "F1-Score":   round(f1_score(y_true, y_pred,        average="weighted", zero_division=0), 6),
        "ROC_AUC":    round(roc_auc_score(y_true, y_prob), 6) if y_prob is not None else "N/A",
    }

    # ── Persist to CSV ──────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"{dataset_name}_metrics.csv")
    df = pd.DataFrame([metrics])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

    # ── Console report ──────────────────────────────────────────────────────────
    separator = "─" * 60
    print(f"\n{separator}")
    print(f"  {model_name} Results on {dataset_name}")
    print(separator)
    print(classification_report(y_true, y_pred, target_names=["Fake (0)", "Genuine (1)"],
                                zero_division=0))
    print(f"  ROC AUC : {metrics['ROC_AUC']}")
    print(separator)

    # ── Confusion Matrix ────────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    print("\n  Confusion Matrix:")
    print(f"  {'':>10}  Pred Fake  Pred Genuine")
    print(f"  {'True Fake':>10}  {cm[0,0]:>9}  {cm[0,1]:>12}")
    print(f"  {'True Gen.':>10}  {cm[1,0]:>9}  {cm[1,1]:>12}\n")

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Genuine"])
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name} on {dataset_name}", fontsize=11, pad=10)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f"{dataset_name}_{model_name}_confusion_matrix.png")
    fig.savefig(cm_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → {cm_path}")

    return metrics


def plot_roc_curve(y_true, y_prob, model_name: str, dataset_name: str) -> None:
    """
    Plot and save the ROC curve at publication quality.
    """
    if y_prob is None:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.fill_between(fpr, tpr, y2=0, alpha=0.08)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name} on {dataset_name}", fontsize=12, pad=10)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, f"{dataset_name}_{model_name}_roc.png")
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curve saved → {save_path}")
