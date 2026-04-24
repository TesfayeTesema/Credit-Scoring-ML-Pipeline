import joblib
import numpy as np
from src.config import MODEL_PATH, SCALER_PATH

def compute_psi(expected, actual, buckets=10):
    """Compute Population Stability Index between two distributions."""
    expected = np.array(expected)
    actual   = np.array(actual)
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual,   bins=breakpoints)[0]
    exp_pct = exp_counts / exp_counts.sum() + 1e-9
    act_pct = act_counts / act_counts.sum() + 1e-9
    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return psi

def run_deployment_monitoring(tuned_model, scaler, X_test, y_test, X_train_raw, X_test_raw, FEATURES, best_eval_name, tuned_auc, df_len):
    """
    STEP 10: Deployment & Monitoring
    """
    print("\n" + "="*65)
    print("STEP 10 ▸ DEPLOYMENT & MONITORING")
    print("="*65)

    print("\n  🚀  Deployment:")
    joblib.dump(tuned_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"      ✅  Model  saved → {MODEL_PATH}")
    print(f"      ✅  Scaler saved → {SCALER_PATH}")

    loaded_model  = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    sample = X_test.iloc[[0]]
    sample_scaled = loaded_scaler.transform(sample)
    prob = loaded_model.predict_proba(sample_scaled)[0, 1]
    risk = "HIGH" if prob > 0.60 else "MEDIUM" if prob > 0.30 else "LOW"
    print(f"\n      Deployment sanity-check on sample row 0:")
    print(f"        P(Distress) = {prob:.4f}   Risk Label = {risk}")
    print(f"        True Label  = {'DISTRESS' if y_test.iloc[0] == 1 else 'NO DISTRESS'}")

    print("\n  📡  Monitoring Strategy:")
    feature_to_monitor = "RevolvingUtilizationOfUnsecuredLines"
    psi_val = compute_psi(
        X_train_raw[feature_to_monitor].dropna().values,
        X_test_raw[feature_to_monitor].dropna().values
    )
    print(f"\n      PSI ({feature_to_monitor}) Train→Test : {psi_val:.4f}")
    stability = "STABLE ✅" if psi_val < 0.10 else "MONITOR ⚠" if psi_val < 0.25 else "RE-TRAIN 🔴"
    print(f"      Stability Status : {stability}")

    print("""
      📋  Monitoring Recommendations for Production:
      ─────────────────────────────────────────────
      1. Schedule monthly PSI checks on all model features.
      2. Log every API prediction to database with timestamp.
      3. Compute rolling 30-day AUC on resolved-loan outcomes.
      4. Trigger automated alert if AUC drops significantly.
      5. Re-train quarterly on the latest data.
    """)

    print("\n" + "="*65)
    print("  🎉  PIPELINE COMPLETE!")
    print("="*65)
    print(f"""
  Summary
  ───────
  Dataset      : cs-training.csv  ({df_len:,} clean rows, {len(FEATURES)} features)
  Best Model   : {best_eval_name}  (Tuned AUC = {tuned_auc:.4f})
  Saved artefacts : {MODEL_PATH} | {SCALER_PATH}
    """)
