import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FEATURES_BASE, TARGET

def run_eda(df):
    """
    STEP 4: Exploratory Data Analysis (EDA)
    Statistics, Visualizations, Find Correlations, Check Balance
    """
    print("\n" + "="*65)
    print("STEP 4 ▸ EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*65)

    # 1. Statistics
    print("\n  📈  Statistics Summary (Clean Data):")
    print(df[FEATURES_BASE].describe().round(2).to_string())

    # 2. Check Balance
    print("\n  ⚖️  Check Balance:")
    print(df[TARGET].value_counts(normalize=True).round(4) * 100)

    # 3. Find Correlations
    print("\n  🔗  Find Correlations (Pearson):")
    corr_matrix = df[FEATURES_BASE + [TARGET]].corr()
    
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix (Pearson)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("03_correlation_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  💾  Saved → 03_correlation_matrix.png")

    target_corr = corr_matrix[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print("\n  Top Feature Correlations with Target:")
    print(target_corr.apply(lambda x: f"{x:+.4f}").to_string())

    # 4. Visualizations
    print("\n  📊  Visualizations: Distributions & Box-plots by Class")
    
    # Distribution plots
    n_cols = 2
    n_rows = (len(FEATURES_BASE) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes_flat = axes.flatten()
    for i, col in enumerate(FEATURES_BASE):
        ax = axes_flat[i]
        for val, label, color in [(0, "No Distress", "#4CAF50"), (1, "Distress", "#E53935")]:
            subset = df[df[TARGET] == val][col].dropna()
            ax.hist(subset, bins=40, alpha=0.55, label=label, color=color, density=True, edgecolor="none")
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle("Feature Distributions by Target Class", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("04_feature_distributions.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  💾  Saved → 04_feature_distributions.png")

    # Box-plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes_flat = axes.flatten()
    for i, col in enumerate(FEATURES_BASE):
        ax = axes_flat[i]
        df.boxplot(column=col, by=TARGET, ax=ax, flierprops=dict(marker=".", alpha=0.3, markersize=2), patch_artist=True)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle("Box-Plots: Features vs Target", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("05_boxplots.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  💾  Saved → 05_boxplots.png")
