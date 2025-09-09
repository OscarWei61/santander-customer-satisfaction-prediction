
# Santander Customer Satisfaction Prediction

This repository contains my solution for the Kaggle "Santander Customer Satisfaction" competition. The project covers data preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and submission generation.

## Objective
Predict whether a Santander bank customer is satisfied (TARGET=1) based on anonymized features, to help the bank improve customer experience and retention.

## Project Pipeline

### 1. Data Loading & Exploration
- Load the raw training data (`train.csv`), check columns, missing values, and target distribution.
- Drop the `ID` column, set `TARGET` as the label `y`, and the rest as features `x`.

### 2. Data Preprocessing
- Split the data into training and validation sets using `train_test_split`.
- Remove features with zero variance.
- Remove highly correlated features (Pearson correlation > 0.8).

### 3. Feature Selection
- Use `ExtraTreesClassifier` to compute feature importances.
- Keep the top 10% most important features and all features with importance > 0; drop the rest.
- Visualize feature importance distribution.

### 4. Modeling & Evaluation
- Try multiple models (e.g., `MLPClassifier`, `XGBoost`).
- Evaluate models using cross-validation (ROC AUC, cv=5).
- Use `GridSearchCV` for MLP hyperparameter tuning.
- Use `RandomizedSearchCV` for XGBoost hyperparameter tuning and apply the best parameters.

### 5. Final Model Training & Prediction
- Train XGBoost with the best parameters on the full training set.
- Predict on the test set and generate `submission.csv` for Kaggle.

## Main Libraries Used
- Python, pandas, numpy, matplotlib, seaborn
- scikit-learn (ExtraTreesClassifier, MLPClassifier, GridSearchCV, cross_val_score)
- imbalanced-learn (SMOTE)
- xgboost (XGBClassifier)

## How to Run
1. Install required packages (see `requirements.txt` or use conda/pip).
2. Run `main.ipynb` step by step.
3. The generated `submission.csv` can be uploaded directly to Kaggle.

## Key Code Snippets

### Feature Importance Selection
```python
from sklearn.ensemble import ExtraTreesClassifier
tree = ExtraTreesClassifier(criterion='entropy')
tree.fit(x_train, y_train)
importances = tree.feature_importances_
# ...sort and select features by importance...
```

### XGBoost Hyperparameter Tuning & Training
```python
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
param_distributions = {
	'booster': ['dart'],
	'n_estimators': [200, 7000],
	'learning_rate': [0.005, 0.05],
	'max_depth': [3, 30],
	'min_child_weight': [3, 10]
}
random_search = RandomizedSearchCV(XGBClassifier(), param_distributions, n_iter=12, cv=3, scoring='roc_auc')
random_search.fit(x_train, y_train)
```

## Results & Notes
- Feature engineering and hyperparameter tuning significantly improved AUC scores.
- Further improvements can be made by trying ensemble methods, feature combinations, or advanced techniques.

For full code and details, please refer to `main.ipynb`.
