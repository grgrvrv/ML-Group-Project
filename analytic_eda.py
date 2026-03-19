import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Combined12.csv')
# generating EDA 
df['temp_range'] = df['temp_max(c)'] - df['temp_min(c)']

print("--- Generating Exploratory Data Analysis (EDA) Charts ---")

# Distribution of Weather Labels (Class Imbalance)
plt.figure(figsize=(7, 4))
sns.countplot(x='normalized_label', data=df, palette='viridis')
plt.title('Distribution of Weather Labels (Class Imbalance)')
plt.show()

# Feature Correlation Heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

# Mean Temperature Distribution across Different Weather Classes
plt.figure(figsize=(8, 5))
sns.boxplot(x='normalized_label', y='temp_mean(c)', data=df, palette='Set2')
plt.title('Mean Temperature Distribution across Different Weather Classes')
plt.show()

print("✅ EDA Visualization Completed")