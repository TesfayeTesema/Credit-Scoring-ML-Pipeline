def run_define_problem():
    """
    STEP 1: Define the Problem
    Identify the goal and define success metrics.
    """
    print("\n" + "="*65)
    print("STEP 1 ▸ DEFINE THE PROBLEM")
    print("="*65)
    
    print("\n  🎯  Goal:")
    print("      Predict whether a borrower will experience 90-days-or-more")
    print("      delinquency within the next 2 years (SeriousDlqin2yrs).")
    
    print("\n  📏  Success Metrics:")
    print("      - High Recall: Prioritize catching potential defaults.")
    print("      - Explainability: Understand model decisions via SHAP.")
    print("      - Robustness: Handle high class imbalance effectively.")
    print("      - Performance: Target ROC-AUC > 0.75.")
    print()
