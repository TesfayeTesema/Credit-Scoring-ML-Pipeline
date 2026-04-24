import numpy as np
import matplotlib.pyplot as plt

def run_data_cleaning_preprocessing(df_raw):
    """
    STEP 3: Data Cleaning & Preprocessing
    Handle Missing Values, Remove Duplicates, Outlier Detection, Data Type Conversion, etc.
    """
    print("\n" + "="*65)
    print("STEP 3 ▸ DATA CLEANING & PREPROCESSING")
    print("="*65)

    df = df_raw.copy()

    # 1. Remove Duplicates
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_after  = len(df)
    print(f"\n  🗑️  Remove Duplicates : {n_before - n_after} removed")

    # 2. Outlier Detection & Handling
    age_zeros = (df["age"] == 0).sum()
    df.loc[df["age"] == 0, "age"] = df["age"].median()
    print(f"  🚨  Outlier Detection : age = 0 fixed ({age_zeros} rows → replaced with median)")

    high_util = (df["RevolvingUtilizationOfUnsecuredLines"] > 1).sum()
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=1.0)
    print(f"  🚨  Outlier Detection : Revolving util > 1 ({high_util} rows capped at 1.0)")

    p99_dr = df["DebtRatio"].quantile(0.99)
    high_dr = (df["DebtRatio"] > p99_dr).sum()
    df["DebtRatio"] = df["DebtRatio"].clip(upper=p99_dr)
    print(f"  🚨  Outlier Detection : DebtRatio > 99th pct ({high_dr} rows capped)")

    delinq_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse"
    ]
    for col in delinq_cols:
        extreme = (df[col] >= 90).sum()
        df[col] = df[col].clip(upper=10)
        print(f"  🚨  Outlier Detection : {col} >= 90 ({extreme} rows capped at 10)")

    # 3. Handle Missing Values
    df["age_decade"] = (df["age"] // 10) * 10
    df["MonthlyIncome"] = df.groupby("age_decade")["MonthlyIncome"].transform(
        lambda x: x.fillna(x.median())
    )
    df["MonthlyIncome"].fillna(df["MonthlyIncome"].median(), inplace=True)
    df.drop(columns=["age_decade"], inplace=True)
    print(f"  🩹  Handle Missing Values: MonthlyIncome imputed with age-decade median")

    df["NumberOfDependents"].fillna(df["NumberOfDependents"].median(), inplace=True)
    print(f"  🩹  Handle Missing Values: NumberOfDependents imputed with median")

    # Final null check
    remaining_nulls = df.isnull().sum().sum()
    print(f"\n  ✅  Total remaining nulls after cleaning : {remaining_nulls}")
    print(f"  📐  Clean dataset shape                  : {df.shape}")

    # 4. Data Transformation (Others according to data)
    print("\n  🔄  Data Transformation: Log-transforming skewed features")
    df["MonthlyIncome_log"] = np.log1p(df["MonthlyIncome"])
    df["DebtRatio_log"]     = np.log1p(df["DebtRatio"])

    # Visualise before/after distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for col, log_col, ax_raw, ax_log in [
        ("MonthlyIncome",  "MonthlyIncome_log",  axes[0][0], axes[0][1]),
        ("DebtRatio",      "DebtRatio_log",       axes[1][0], axes[1][1])
    ]:
        df[col].hist(bins=60, ax=ax_raw, color="#42A5F5", edgecolor="white")
        ax_raw.set_title(f"{col}  (original)")
        df[log_col].hist(bins=60, ax=ax_log, color="#26A69A", edgecolor="white")
        ax_log.set_title(f"{log_col}  (log1p)")
    plt.tight_layout()
    plt.savefig("02_log_transform.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  💾  Saved → 02_log_transform.png")

    return df
