import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File paths
DATA_PATH = os.path.join(BASE_DIR, "cs-training.csv")
MODEL_PATH = os.path.join(BASE_DIR, "credit_scoring_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "credit_scoring_scaler.pkl")

# Random state for reproducibility
RANDOM_STATE = 42

# Target variable
TARGET = "SeriousDlqin2yrs"

# Base features after log transformation
FEATURES_BASE = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio_log",
    "MonthlyIncome_log", "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"
]
