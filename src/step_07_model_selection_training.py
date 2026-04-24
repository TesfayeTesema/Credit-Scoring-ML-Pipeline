import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.config import RANDOM_STATE

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

def run_model_selection_training(X_train_res, y_train_res, y_train):
    """
    STEP 7: Model Selection & Training
    Choose Algorithms, Training
    """
    print("\n" + "="*65)
    print("STEP 7 ▸ MODEL SELECTION & TRAINING")
    print("="*65)

    # 1. Choose Algorithms
    print("\n  🧠  Choose Algorithms:")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", n_jobs=-1, scale_pos_weight=(y_train==0).sum() / (y_train==1).sum(), random_state=RANDOM_STATE)
    if LGB_AVAILABLE:
        models["LightGBM"] = lgb.LGBMClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
    
    for name in models.keys():
        print(f"      - {name}")

    # 2. Model Selection via CV
    print("\n  🔬  Model Selection (5-Fold CV ROC-AUC):")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = scores
        print(f"      {name:<25}  AUC = {scores.mean():.4f}  ± {scores.std():.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    names  = list(cv_results.keys())
    means  = [v.mean() for v in cv_results.values()]
    stds   = [v.std()  for v in cv_results.values()]
    ax.barh(names, means, xerr=stds, color="#5C6BC0", edgecolor="black", height=0.5, capsize=4)
    ax.set_xlabel("CV ROC-AUC")
    ax.set_title("Model Comparison – 5-Fold Cross-Validation ROC-AUC", fontweight="bold")
    ax.axvline(x=0.75, color="red", linestyle="--", label="Target ≥ 0.75")
    ax.legend()
    plt.tight_layout()
    plt.savefig("06_model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("\n  💾  Saved → 06_model_comparison.png")

    best_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
    print(f"\n  🏆  Best algorithm by CV AUC : {best_model_name}  ({cv_results[best_model_name].mean():.4f})")

    # 3. Training All Models
    print("\n  ⚙️  Training Models on Full Training Set:")
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        trained_models[name] = model
        print(f"      ✅  Trained : {name}")

    return trained_models, best_model_name
