import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and clean data
df = pd.read_csv('Combined12.csv').dropna()

# Feature engineering
df['temp_range'] = df['temp_max(c)'] - df['temp_min(c)']
print("Added new feature: 'temp_range' (Daily Temperature Range)")

X = df.drop('normalized_label', axis=1)
y = df['normalized_label']

# Stratified sampling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature standardization
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Save core data
X_train_scaled.to_csv('X_train_scaled.csv', index=False)
X_test_scaled.to_csv('X_test_scaled.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print("\n✅ Data splitting and standardization completed! Core feature files transferred.")