import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve

def run_evaluation(X_test_scaled, y_test, trained_models, best_eval_name):
    """
    STEP 8: Evaluation
    Classification metrics (Accuracy, Precision, Recall, F1), Overfitting Check
    """
    print("\n" + "="*65)
    print("STEP 8 ▸ EVALUATION")
    print("="*65)

    eval_rows = []
    fig_roc, ax_roc = plt.subplots(figsize=(9, 6))
    ax_roc.plot([0,1],[0,1], "k--", label="Random Guesser (AUC=0.50)")

    print("\n  📊  Evaluating Models on Test Set...")

    for name, model in trained_models.items():
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_prob)
        
        eval_rows.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC-AUC": auc})
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})")

    eval_df = pd.DataFrame(eval_rows).set_index("Model").sort_values("ROC-AUC", ascending=False)
    
    print("\n  Test-Set Metrics (Classification):")
    print(eval_df.round(4).to_string())

    # Save ROC curves
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves – All Models", fontweight="bold")
    ax_roc.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("07_roc_curves.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("\n  💾  Saved → 07_roc_curves.png")

    best_eval_name = eval_df["ROC-AUC"].idxmax()
    best_eval_model = trained_models[best_eval_name]
    
    # Overfitting Check (Conceptual text based on CV vs Test)
    print("\n  🛡️  Overfitting Check:")
    print(f"      Best Model ({best_eval_name}) Test AUC = {eval_df.loc[best_eval_name, 'ROC-AUC']:.4f}")
    print("      (Compare this to CV AUC from Step 7. If Test is significantly lower, model is overfitting).")

    # Detailed metrics for best model
    y_pred_best = best_eval_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Distress (0)", "Distress (1)"])
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(f"Confusion Matrix – {best_eval_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig("08_confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  💾  Saved → 08_confusion_matrix.png")

    print(f"\n  Classification Report – {best_eval_name}:")
    print(classification_report(y_test, y_pred_best, target_names=["No Distress", "Distress"]))

    return eval_df, best_eval_name
