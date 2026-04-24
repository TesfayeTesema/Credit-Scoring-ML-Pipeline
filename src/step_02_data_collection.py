import pandas as pd
import matplotlib.pyplot as plt
import os
from src.config import DATA_PATH

def run_data_collection():
    """
    STEP 2: Data Collection
    Sources, Quantity vs. Quality
    """
    print("\n" + "="*65)
    print("STEP 2 ▸ DATA COLLECTION")
    print("="*65)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    # Load raw data
    print("\n  📥  Loading data from Source...")
    df_raw = pd.read_csv(DATA_PATH, index_col=0)

    print(f"\n  📂  Dataset loaded from  : {DATA_PATH}")
    print(f"  📐  Quantity (Shape)     : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    
    print("\n  🔍  Quality Check (Missing Values):\n")
    missing = df_raw.isnull().sum()
    missing_pct = (df_raw.isnull().sum() / len(df_raw) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    missing_df = missing_df[missing_df["Missing Count"] > 0]
    print(missing_df.to_string() if not missing_df.empty else "  → No missing values found!")

    print("\n\n  🎯  Target Class Distribution:\n")
    target_counts = df_raw["SeriousDlqin2yrs"].value_counts()
    target_pct    = df_raw["SeriousDlqin2yrs"].value_counts(normalize=True) * 100
    print(pd.DataFrame({"Count": target_counts, "Percent (%)": target_pct.round(2)}))
    print(f"\n  Imbalance Ratio  ≈  1 : {target_counts[0] / target_counts[1]:.1f}  (Non-distress : Distress)")
    print("  ⚠  High class imbalance detected – will address later in pipeline.")

    # Visualisation – Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    target_counts.plot(kind="bar", ax=axes[0], color=["#4CAF50", "#E53935"], edgecolor="black")
    axes[0].set_title("Target Class Distribution (Count)")
    axes[0].set_xlabel("SeriousDlqin2yrs  (0=No Distress, 1=Distress)")
    axes[0].set_ylabel("Count")
    axes[0].set_xticklabels(["No Distress (0)", "Distress (1)"], rotation=0)
    
    target_pct.plot(kind="pie", ax=axes[1], autopct="%1.1f%%", colors=["#4CAF50", "#E53935"], startangle=90)
    axes[1].set_ylabel("")
    axes[1].set_title("Target Class Distribution (%)")
    
    plt.tight_layout()
    plt.savefig("01_target_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  💾  Saved → 01_target_distribution.png")

    return df_raw
