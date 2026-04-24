import numpy as np
import pandas as pd
from src.config import FEATURES_BASE, TARGET

def run_feature_engineering(df):
    """
    STEP 5: Feature Engineering & Selection
    Feature Scaling, Encoding, Feature Creation, Selection
    """
    print("\n" + "="*65)
    print("STEP 5 ▸ FEATURE ENGINEERING & SELECTION")
    print("="*65)

    df_fe = df.copy()

    # 1. Feature Creation
    print("\n  🛠️  Feature Creation:")
    
    df_fe["TotalDelinquencies"] = (
        df_fe["NumberOfTime30-59DaysPastDueNotWorse"] +
        df_fe["NumberOfTimes90DaysLate"] +
        df_fe["NumberOfTime60-89DaysPastDueNotWorse"]
    )
    print("      ✅  TotalDelinquencies created")

    df_fe["DelinquencyRate"] = df_fe["TotalDelinquencies"] / (0.01 + df_fe["MonthlyIncome_log"])
    print("      ✅  DelinquencyRate created")

    df_fe["IncomePerDependent"] = df_fe["MonthlyIncome"] / (1 + df_fe["NumberOfDependents"])
    df_fe["IncomePerDependent_log"] = np.log1p(df_fe["IncomePerDependent"])
    print("      ✅  IncomePerDependent_log created")

    # 2. Encoding / Binning
    print("\n  🔢  Encoding & Binning:")
    
    df_fe["CreditUtilBin"] = pd.cut(
        df_fe["RevolvingUtilizationOfUnsecuredLines"],
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)
    print("      ✅  CreditUtilBin (0-4) created")

    df_fe["AgeGroup"] = pd.cut(
        df_fe["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(float)
    print("      ✅  AgeGroup (0-5) created")

    # 3. Feature Selection
    print("\n  🧲  Feature Selection:")
    FEATURES = FEATURES_BASE + [
        "TotalDelinquencies", "DelinquencyRate",
        "IncomePerDependent_log", "CreditUtilBin", "AgeGroup"
    ]
    print(f"      ✅  Selected Features : {len(FEATURES)}")

    new_feats = ["TotalDelinquencies", "DelinquencyRate", "IncomePerDependent_log", "CreditUtilBin", "AgeGroup"]
    print("\n  Correlation of newly engineered features with target:")
    print(df_fe[new_feats + [TARGET]].corr()[TARGET].drop(TARGET).apply(lambda x: f"{x:+.4f}").to_string())

    return df_fe, FEATURES
