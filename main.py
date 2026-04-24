import warnings
import seaborn as sns

# Ignore warnings globally
warnings.filterwarnings("ignore")

# Global seaborn style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

from src.step_01_define_problem import run_define_problem
from src.step_02_data_collection import run_data_collection
from src.step_03_data_cleaning_preprocessing import run_data_cleaning_preprocessing
from src.step_04_eda import run_eda
from src.step_05_feature_engineering_selection import run_feature_engineering
from src.step_06_data_splitting import run_data_splitting
from src.step_07_model_selection_training import run_model_selection_training
from src.step_08_evaluation import run_evaluation
from src.step_09_hyperparameter_tuning import run_hyperparameter_tuning
from src.step_10_deployment_monitoring import run_deployment_monitoring

def main():
    print("============================================================")
    print(" CREDIT SCORING – Full ML Pipeline (10 Steps) ")
    print("============================================================")

    # 1. Define the Problem
    run_define_problem()

    # 2. Data Collection
    df_raw = run_data_collection()

    # 3. Data Cleaning & Preprocessing
    df_clean = run_data_cleaning_preprocessing(df_raw)

    # 4. Exploratory Data Analysis (EDA)
    run_eda(df_clean)

    # 5. Feature Engineering & Selection
    df_fe, FEATURES = run_feature_engineering(df_clean)
    
    df_len = len(df_fe)

    # 6. Data Splitting
    X_train_res, y_train_res, X_test_scaled, y_test, X_train_raw, X_test_raw, scaler, y_train = run_data_splitting(df_fe, FEATURES)

    # 7. Model Selection & Training
    trained_models, best_model_name = run_model_selection_training(X_train_res, y_train_res, y_train)

    # 8. Evaluation
    eval_df, best_eval_name = run_evaluation(X_test_scaled, y_test, trained_models, best_model_name)

    # 9. Hyperparameter Tuning
    tuned_model, tuned_auc = run_hyperparameter_tuning(
        X_train_res, y_train_res, X_test_scaled, y_test, X_test_raw, FEATURES, best_eval_name, eval_df, y_train
    )

    # 10. Deployment & Monitoring
    run_deployment_monitoring(
        tuned_model, scaler, X_test_raw, y_test, 
        X_train_raw, X_test_raw, FEATURES, best_eval_name, tuned_auc, df_len
    )

if __name__ == "__main__":
    main()
