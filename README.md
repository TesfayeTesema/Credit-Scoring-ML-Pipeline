# CreditGuard AI: End-to-End Credit Scoring Pipeline
Predicting financial distress with explainable AI to support data-driven lending decisions.

 Project Overview
In the context of inclusive finance (specifically modeled for Ahadu Bank), this project implements a full-scale Machine Learning lifecycle to predict whether a borrower will experience 90-days-or-more delinquency within the next two years.
By utilizing historical bureau data of 150,000 borrowers, this pipeline transforms raw, "noisy" data into a production-ready model capable of identifying high-risk individuals while providing clear explanations for every decision.

## Business Objective
In the context of inclusive finance, accurately assessing credit risk is crucial. This pipeline provides:
- **High Recall**: Prioritizes catching potential defaults.
- **Explainability**: Uses SHAP values to understand *why* a model made a specific decision.
- **Robustness**: Handles high class imbalance using SMOTE and algorithm-level class weighting.

## The 10-Step Pipeline Architecture
-The project is strictly modularized into the standard ML lifecycle:
- Define Problem : Goal setting and success metric (ROC-AUC > 0.75) definition.
- Data Collection: Auditing 150k records and identifying high class imbalance.
- Data Cleaning: Smart imputation (Age-Decade Medians) and outlier capping.
- **EDA: Statistical correlation analysis and feature relationship mapping.
- **Feature Engineering: Creating synthetic features like TotalDelinquencies and DisposableIncome.
- **Data Splitting : Stratified splitting and SMOTE balancing of the training set.
- **Model Selection: Comparative analysis of Logistic Regression, Random Forest, and XGBoost.
- **Evaluation: Validating performance using Recall, F1-Score, and ROC Curves.
- **Tuning & SHAP: Hyperparameter optimization and Explainable AI (XAI) logic.
- **Deployment & Monitoring: Model serialization and Population Stability Index (PSI) tracking.


##Key Technical Highlights
 Explainable AI (SHAP)
   Unlike "black-box" models, we use SHAP (SHapley Additive exPlanations) to visualize exactly why a loan was           denied.   This ensures the model is fair, transparent, and compliant with banking regulations.
## Handling Class Imbalance
With only 6.7% of borrowers in the "Distress" class, standard models fail. We implemented SMOTE (Synthetic Minority Over-sampling Technique) to synthesize new minority examples, ensuring the model prioritizes catching defaults (High Recall).
## Feature Engineering Power
By creating the TotalDelinquencies score, we increased our primary risk signal by 10% compared to using raw bureau counts alone.

##Installation & Usage
1. Clone the repository

git clone https://github.com/yourusername/credit-scoring-pipeline.git
cd credit-scoring-pipeline
2. Install Dependencies

pip install -r requirements.txt
3. Run the Pipeline

python main.py


## Project Structure
The pipeline is strictly structured into the standard 10-step Machine Learning Lifecycle:

```text
├── main.py                                      # Main orchestrator
├── README.md                                    # Project documentation
├── cs-training.csv                              # Raw dataset (must be present)
├── src/
│   ├── config.py                                # Configuration, constants, and paths
│   ├── step_01_define_problem.py                # Identifies goals and success metrics
│   ├── step_02_data_collection.py               # Data loading and initial checks
│   ├── step_03_data_cleaning_preprocessing.py   # Missing values, outliers, log-transforms
│   ├── step_04_eda.py                           # Statistics, correlations, visuals
│   ├── step_05_feature_engineering_selection.py # Feature creation, encoding, scaling
│   ├── step_06_data_splitting.py                # Train/test split, SMOTE balancing
│   ├── step_07_model_selection_training.py      # Cross-validation and algorithm training
│   ├── step_08_evaluation.py                    # Classification metrics, ROC-AUC
│   ├── step_09_hyperparameter_tuning.py         # RandomizedSearchCV and SHAP explainability
│   └── step_10_deployment_monitoring.py         # Serialization and PSI monitoring strategy
```

## Setup & Installation

1. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm shap joblib
   ```
2. **Ensure data is present**: 
   Ensure `cs-training.csv` is located in the root directory.

## Execution Guide

To run the full 10-step pipeline, execute the main orchestrator script:
```bash
python main.py
```
This script will sequentially run through all 10 steps from problem definition to deployment and monitoring recommendations.

### Expected Outputs
- **Plots**: Generates and saves multiple plots (e.g., `01_target_distribution.png`, `07_roc_curves.png`, `10_shap_beeswarm.png`).
- **Artifacts**: Serialized models (`credit_scoring_model.pkl`) and scalers (`credit_scoring_scaler.pkl`) are saved for production.
- **Terminal Output**: Step-by-step interpretation logging cleaning decisions, metric calculations, and evaluations.

## Pipeline Steps Details

1. **Define the Problem**: Set goals (predict distress) and success metrics (high recall, AUC > 0.75).
2. **Data Collection**: Load data, assess quantity vs. quality, observe class imbalance.
3. **Data Cleaning & Preprocessing**: Impute missing values with medians, cap outliers, and apply log-transformations.
4. **Exploratory Data Analysis (EDA)**: Calculate correlations, plot distributions, and assess balance.
5. **Feature Engineering & Selection**: Create `TotalDelinquencies`, `DelinquencyRate`, `IncomePerDependent`.
6. **Data Splitting**: 80/20 train-test split, standard scaling, and applying SMOTE to balance the training set.
7. **Model Selection & Training**: Evaluate Logistic Regression, Random Forest, Gradient Boosting, XGBoost, etc. via 5-fold CV.
8. **Evaluation**: Assess on Test Set using Accuracy, Precision, Recall, F1, and AUC. Check for overfitting.
9. **Hyperparameter Tuning**: Tune the best model using `RandomizedSearchCV`. Explain the model using `SHAP` values.
10. **Deployment & Monitoring**: Save artifacts. Define a monitoring strategy including Population Stability Index (PSI).
