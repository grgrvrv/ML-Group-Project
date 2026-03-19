import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

X_train = pd.read_csv('X_train_scaled.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
X_test = pd.read_csv('X_test_scaled.csv')

# 1. Train Logistic Regression (Handling multi-class and imbalance)
log_model = LogisticRegression(max_iter=1000, multi_class='multinomial', class_weight='balanced')

# Demonstrating rigor: Perform 5-fold cross-validation
print("Executing 5-Fold cross-validation to evaluate the stability of Logistic Regression...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(log_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})\n")

# 2. Finally fit the baseline models and predict
log_model.fit(X_train, y_train)
dt_model = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
dt_model.fit(X_train, y_train)

# Save prediction results
pd.DataFrame(log_model.predict(X_test), columns=['pred']).to_csv('log_preds.csv', index=False)
pd.DataFrame(dt_model.predict(X_test), columns=['pred']).to_csv('dt_preds.csv', index=False)
print("✅ Baseline models (Logistic & Decision Tree) prediction completed, results handed over!")