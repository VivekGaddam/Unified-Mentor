import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('bird_strike_data.csv')

# Data Preprocessing
## Handling missing values
df.fillna(method='ffill', inplace=True)

## Data transformation (if necessary)
scaler = StandardScaler()
df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])

# Exploratory Data Analysis (EDA)
## Distribution of data
plt.figure(figsize=(10, 6))
sns.histplot(df['feature'], kde=True)
plt.title('Distribution of Feature')
plt.show()

## Relationships between variables
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='feature1', y='feature2', hue='target')
plt.title('Feature1 vs Feature2')
plt.show()

# Feature Engineering
## Creating new features
df['new_feature'] = df['feature1'] * df['feature2']

# Model Building and Evaluation
## Split data into train and test sets
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

## Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Visualization
## Feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.show()
