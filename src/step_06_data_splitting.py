from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import TARGET, RANDOM_STATE

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠  imbalanced-learn not installed – SMOTE step will be skipped.")

def run_data_splitting(df, FEATURES):
    """
    STEP 6: Data Splitting
    Training Set, Validation Set (handled in CV), Test Set
    """
    print("\n" + "="*65)
    print("STEP 6 ▸ DATA SPLITTING")
    print("="*65)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Fill any remaining NaN from engineered features
    X = X.fillna(X.median())

    # 1. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print("\n  ✂️  Data Splitting:")
    print(f"      Training set   : {X_train.shape[0]:,} rows | Class 0: {(y_train==0).sum():,} | Class 1: {(y_train==1).sum():,}")
    print(f"      Test set       : {X_test.shape[0]:,} rows  | Class 0: {(y_test==0).sum():,}  | Class 1: {(y_test==1).sum():,}")

    # 2. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("\n  📏  Scaling:")
    print("      ✅  StandardScaler fitted on training set and applied to both splits.")

    # 3. Handle Imbalance (SMOTE)
    if SMOTE_AVAILABLE:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        print(f"\n  ⚖️  SMOTE applied on Training Set:")
        print(f"      Before → Class 0: {(y_train==0).sum():,}  Class 1: {(y_train==1).sum():,}")
        print(f"      After  → Class 0: {(y_train_res==0).sum():,}  Class 1: {(y_train_res==1).sum():,}")
    else:
        X_train_res, y_train_res = X_train_scaled, y_train
        print("\n  ⚠  SMOTE skipped – using original imbalanced training set.")

    return X_train_res, y_train_res, X_test_scaled, y_test, X_train, X_test, scaler, y_train
