import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, auc, matthews_corrcoef, balanced_accuracy_score
)
import config as _config

_DPI = 150

def evaluate_pad_model(
    y_true, y_pred, y_prob,
    model_name: str,
    experiment_name: str,
    feature_dim: int,
    train_genuine: int,
    train_fake: int,
    best_params: str,
    cv_roc_auc: float
) -> dict:
    """
    Compute comprehensive PAD metrics and save confusion matrices, ROC curves,
    and PR curves to results/<experiment_name>/.
    """
    # Read RESULTS_DIR at call time so any runtime override (e.g. from notebook) takes effect
    RESULTS_DIR = _config.RESULTS_DIR
    exp_dir = os.path.join(RESULTS_DIR, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # ── Core Metrics ──
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision) if y_prob is not None else 0.0
    
    # Class-specific (0=Fake, 1=Genuine)
    fake_prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    fake_rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    fake_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    gen_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    gen_rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    gen_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # ── PAD Metrics (FAR & FRR) ──
    # True Negative (TN): Actual Fake (0), Predicted Fake (0)
    # False Positive (FP): Actual Fake (0), Predicted Genuine (1) -> False Acceptance
    # False Negative (FN): Actual Genuine (1), Predicted Fake (0) -> False Rejection
    # True Positive (TP): Actual Genuine (1), Predicted Genuine (1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    metrics_dict = {
        "Experiment": experiment_name,
        "Model": model_name,
        "TrainSamples": train_genuine + train_fake,
        "TestSamples": len(y_true),
        "TrainGenuine": train_genuine,
        "TrainFake": train_fake,
        "TestGenuine": int((y_true == 1).sum()),
        "TestFake": int((y_true == 0).sum()),
        "FeatureDimension": feature_dim,
        "BestParams": str(best_params),
        "CV_ROC_AUC": round(cv_roc_auc, 6),
        "ROC_AUC": round(roc_auc, 6),
        "PR_AUC": round(pr_auc, 6),
        "Accuracy": round(acc, 6),
        "BalancedAccuracy": round(bal_acc, 6),
        "FakePrecision": round(fake_prec, 6),
        "FakeRecall": round(fake_rec, 6),
        "FakeF1": round(fake_f1, 6),
        "GenuinePrecision": round(gen_prec, 6),
        "GenuineRecall": round(gen_rec, 6),
        "GenuineF1": round(gen_f1, 6),
        "MacroF1": round(macro_f1, 6),
        "MCC": round(mcc, 6),
        "FalseAcceptanceRate": round(far, 6),
        "FalseRejectionRate": round(frr, 6)
    }
    
    # ── Output to Console ──
    test_total  = int((y_true == 1).sum()) + int((y_true == 0).sum())
    test_gen    = int((y_true == 1).sum())
    test_fake   = int((y_true == 0).sum())
    train_total = train_genuine + train_fake

    W = 64
    print(f"\n{'═'*W}")
    print(f"  {model_name} RESULTS  ·  Experiment: {experiment_name.upper()}")
    print(f"{'═'*W}")

    # ── Dataset size block ──
    print(f"  {'DATASET SIZES':}")
    print(f"    Training  : {train_total:>6} images  ({train_genuine} genuine  +  {train_fake} fake)  [balanced undersampled]")
    print(f"    Test      : {test_total:>6} images  ({test_gen} genuine  +  {test_fake} fake)  [natural imbalance]")
    print(f"    Features  : {feature_dim}-dim LBP histogram  (8×8 blocks × 10 bins)")

    # ── Confusion matrix text block ──
    print(f"\n  {'CONFUSION MATRIX  (rows=Actual, cols=Predicted)':}")
    print(f"    {'':20s}  {'Pred Fake':>10}  {'Pred Genuine':>13}  {'Row Total':>10}")
    print(f"    {'Actual Fake  ':20s}  {tn:>10}  {fp:>13}  {tn+fp:>10}")
    print(f"    {'Actual Genuine':20s}  {fn:>10}  {tp:>13}  {fn+tp:>10}")
    print(f"    {'Col Total':20s}  {tn+fn:>10}  {fp+tp:>13}  {tn+fp+fn+tp:>10}  ← sum = {test_total}")

    # ── Metrics table ──
    print(f"\n  {'METRICS':}")
    print(f"    {'Metric':<28} {'Fake (0)':>10}  {'Genuine (1)':>12}  {'Overall':>10}")
    print(f"    {'-'*64}")
    print(f"    {'Precision':<28} {fake_prec:>10.4f}  {gen_prec:>12.4f}")
    print(f"    {'Recall':<28} {fake_rec:>10.4f}  {gen_rec:>12.4f}")
    print(f"    {'F1-Score':<28} {fake_f1:>10.4f}  {gen_f1:>12.4f}")
    print(f"    {'-'*64}")
    print(f"    {'Accuracy':<28} {'':>10}  {'':>12}  {acc:>10.4f}")
    print(f"    {'Balanced Accuracy':<28} {'':>10}  {'':>12}  {bal_acc:>10.4f}")
    print(f"    {'Macro F1':<28} {'':>10}  {'':>12}  {macro_f1:>10.4f}")
    print(f"    {'MCC':<28} {'':>10}  {'':>12}  {mcc:>10.4f}")
    print(f"    {'ROC-AUC  (test)':<28} {'':>10}  {'':>12}  {roc_auc:>10.4f}")
    print(f"    {'PR-AUC   (test)':<28} {'':>10}  {'':>12}  {pr_auc:>10.4f}")
    print(f"    {'CV ROC-AUC (train)':<28} {'':>10}  {'':>12}  {cv_roc_auc:>10.4f}")
    print(f"    {'-'*64}")
    print(f"    {'FAR (False Acceptance Rate)':<28} {'':>10}  {'':>12}  {far:>10.4f}")
    print(f"    {'FRR (False Rejection Rate)':<28} {'':>10}  {'':>12}  {frr:>10.4f}")
    print(f"{'═'*W}\n")

    
    # ── Confusion Matrix Plot ──
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Genuine"])
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name} on {experiment_name}", fontsize=11, pad=10)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    plt.tight_layout()
    cm_path = os.path.join(exp_dir, f"{model_name}_confusion_matrix.png")
    fig.savefig(cm_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    
    # ── ROC Curve Plot ──
    if y_prob is not None:
        fpr_vals, tpr_vals, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr_vals, tpr_vals, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.fill_between(fpr_vals, tpr_vals, y2=0, alpha=0.08)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve — {model_name} on {experiment_name}", fontsize=12)
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        roc_path = os.path.join(exp_dir, f"{model_name}_roc.png")
        fig.savefig(roc_path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)

    # ── PR Curve Plot ──
    if y_prob is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, lw=2, label=f"{model_name} (PR-AUC = {pr_auc:.4f})")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Precision-Recall Curve — {model_name} on {experiment_name}", fontsize=12)
        ax.legend(loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        pr_path = os.path.join(exp_dir, f"{model_name}_pr_curve.png")
        fig.savefig(pr_path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)

    return metrics_dict
