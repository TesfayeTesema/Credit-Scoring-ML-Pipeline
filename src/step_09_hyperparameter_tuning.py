import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from src.config import RANDOM_STATE

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠  SHAP not installed – explainability cell will be skipped.")

def run_hyperparameter_tuning(X_train_res, y_train_res, X_test_scaled, y_test, X_test, FEATURES, best_eval_name, eval_df, y_train):
    """
    STEP 9: Hyperparameter Tuning
    Optimising the best model's parameters and SHAP explainability.
    """
    print("\n" + "="*65)
    print("STEP 9 ▸ HYPERPARAMETER TUNING")
    print("="*65)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n  🎛️  Tuning Hyperparameters for {best_eval_name}...")

    if "Random Forest" in best_eval_name or best_eval_name == "Random Forest":
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }
        base_model = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
    elif "Gradient Boosting" in best_eval_name:
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
            "min_samples_leaf": [1, 5, 10],
        }
        base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    elif XGB_AVAILABLE and "XGBoost" in best_eval_name:
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        }
        base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=-1, scale_pos_weight=(y_train==0).sum() / (y_train==1).sum(), random_state=RANDOM_STATE)
    else:
        param_dist = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
        }
        base_model = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)

    random_search = RandomizedSearchCV(estimator=base_model, param_distributions=param_dist, n_iter=20, scoring="roc_auc", cv=cv, n_jobs=-1, random_state=RANDOM_STATE, verbose=0)
    random_search.fit(X_train_res, y_train_res)

    print(f"\n  Best params   : {random_search.best_params_}")
    print(f"  Best CV AUC   : {random_search.best_score_:.4f}")

    tuned_model = random_search.best_estimator_
    y_pred_tuned = tuned_model.predict(X_test_scaled)
    y_prob_tuned = tuned_model.predict_proba(X_test_scaled)[:, 1]

    tuned_auc  = roc_auc_score(y_test, y_prob_tuned)
    tuned_f1   = f1_score(y_test, y_pred_tuned, zero_division=0)
    tuned_rec  = recall_score(y_test, y_pred_tuned, zero_division=0)
    base_auc   = eval_df.loc[best_eval_name, "ROC-AUC"]

    print(f"\n  ─── Tuned Model Test-Set Results ───")
    print(f"  ROC-AUC  : {tuned_auc:.4f}  (baseline: {base_auc:.4f}, Δ = {tuned_auc - base_auc:+.4f})")
    print(f"  F1-Score : {tuned_f1:.4f}")
    print(f"  Recall   : {tuned_rec:.4f}")

    # Feature Importance & SHAP Explainability
    print("\n  💡  Explainability:")
    if hasattr(tuned_model, "feature_importances_"):
        importances = pd.Series(tuned_model.feature_importances_, index=FEATURES)
        importances_sorted = importances.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(9, 7))
        importances_sorted.plot(kind="barh", ax=ax, color="#EF5350", edgecolor="black")
        ax.set_title("Feature Importances (Built-in)", fontweight="bold")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig("09_feature_importance.png", dpi=120, bbox_inches="tight")
        plt.close()
        print("      💾  Saved → 09_feature_importance.png")

    if SHAP_AVAILABLE:
        print("\n      Computing SHAP values (this may take ~30 seconds) …")
        explainer = shap.Explainer(tuned_model, X_train_res, feature_names=FEATURES)
        X_test_sample = pd.DataFrame(X_test_scaled[:500], columns=FEATURES)
        shap_values   = explainer(X_test_sample)
        fig_shap = plt.figure(figsize=(10, 7))
        shap.plots.beeswarm(shap_values[:, :, 1], max_display=15, show=False)
        plt.title("SHAP Beeswarm – Feature Impact on Distress Prediction", fontweight="bold")
        plt.tight_layout()
        plt.savefig("10_shap_beeswarm.png", dpi=120, bbox_inches="tight")
        plt.close()
        print("      💾  Saved → 10_shap_beeswarm.png")
    else:
        print("      ⚠  SHAP not available – skipping beeswarm plot.")

    return tuned_model, tuned_auc
