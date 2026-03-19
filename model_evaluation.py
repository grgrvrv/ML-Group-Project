import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

y_test = pd.read_csv('y_test.csv').values.ravel()

# Collect model prediction files
preds = {
    'Logistic Regression': pd.read_csv('log_preds.csv').values.ravel(),
    'Decision Tree': pd.read_csv('dt_preds.csv').values.ravel(),
    'Random Forest': pd.read_csv('rf_preds.csv').values.ravel(),
    'LightGBM': pd.read_csv('lgb_preds.csv').values.ravel()
}

# 1. Calculate core metrics (Accuracy and Macro F1)
results = []
for name, y_pred in preds.items():
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Macro F1': f1_score(y_test, y_pred, average='macro')
    })

print("📊 Horizontal comparison of model multi-class performance:")
print(pd.DataFrame(results).sort_values(by='Macro F1', ascending=False).to_markdown(index=False))

for name, y_pred in preds.items():
    best_preds = preds['LightGBM']
    print(f"\n--- Detailed Classification Report for {name} ---")
    print(classification_report(y_test, best_preds))

    # 3. Plot confusion matrix and display on screen (do not save)
    plt.figure(figsize=(7, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('Actual True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print("\n ")