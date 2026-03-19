# Meteorological Data Analysis and Weather Classification Prediction Project

This project is an end-to-end machine learning pipeline covering everything from exploratory data analysis (EDA) of raw meteorological data and feature engineering to the training and evaluation of various machine learning models (Logistic Regression, Decision Tree, Random Forest, LightGBM).

## 📂 Project Structure

* `analytic_eda.py`: Exploratory data analysis, generating class distributions, correlation heatmaps, and feature boxplots.
* `data_cleaning.py`: Data preprocessing script, including missing value handling, feature engineering (calculating daily temperature range), and stratified sampling.
* `lr_dt.py`: Baseline model training, including Logistic Regression (5-fold cross-validation) and Decision Tree.
* `rf_lgbm.py`: Advanced ensemble model training, including manual hyperparameter tuning for Random Forest and the LightGBM model.
* `model_evaluation.py`: Horizontal comparison of model performance, generating classification reports and confusion matrices.

## 🚀 Quick Start

### 1. Environment Preparation
Ensure the following libraries are installed in your Python environment:

```bash
pip install pandas matplotlib seaborn scikit-learn lightgbm

### 2. Execution Flow
Please execute the scripts in the following order:

* **Data Cleaning**: `python data_cleaning.py` (Generates standardized training and testing sets)
* **Data Analysis**: `python analytic_eda.py` (Visualizes feature distributions)
* **Train Baseline Models**: `python lr_dt.py`
* **Train Ensemble Models**: `python rf_lgbm.py`
* **Model Evaluation**: `python model_evaluation.py`

## 📊 Core Technical Points
* **Feature Engineering**: Constructed the `temp_range` (daily temperature range) feature.
* **Class Imbalance Handling**: All models utilize the `class_weight='balanced'` parameter.
* **Rigorous Evaluation**: Uses `StratifiedKFold` for 5-fold cross-validation to ensure model stability.
* **Model Comparison**: Horizontally compares the Macro F1 scores of linear models, tree-based models, and Boosting algorithms.