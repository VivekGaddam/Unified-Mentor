import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("Crop Production data.csv")

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Handle missing values
data = data.dropna()  # or use data.fillna() for imputation

# Convert categorical columns to numerical
data['State_Name'] = data['State_Name'].astype('category').cat.codes
data['District_Name'] = data['District_Name'].astype('category').cat.codes
data['Season'] = data['Season'].astype('category').cat.codes
data['Crop'] = data['Crop'].astype('category').cat.codes

# Display a sample of the data
print(data.sample(10))

# Group by State and plot total production
state_wise = data.groupby(data['State_Name'])["Production"].sum()
state_wise.plot(kind='bar', figsize=(10, 10))
plt.title('Total Production by State')
plt.xlabel('State')
plt.ylabel('Production')
plt.show()

# Line plot of crop production over years
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Crop_Year', y='Production', hue='Crop')
plt.title('Crop Production Over Years')
plt.xlabel('Year')
plt.ylabel('Production')
plt.show()

# Histogram of crop production
plt.figure(figsize=(8, 5))
sns.histplot(data['Production'], bins=30, kde=True)
plt.title('Distribution of Crop Production')
plt.xlabel('Production')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Area cultivated by season
season_area = data[['Season', 'Area']].groupby('Season', as_index=False).sum()
season_area = season_area[season_area['Season'].str.strip() != 'Whole Year']
season_area.plot(kind='barh', x='Season', y='Area', legend=False)
plt.ylabel('Season')
plt.xlabel('Area Cultivated')
plt.title('Seasons vs Area Cultivated')
plt.show()

# Top 5 crops by production
crops = data[['Crop', 'Production']].groupby('Crop').sum()
ordered_crops = crops.sort_values(by='Production', ascending=False)
top5_crops = ordered_crops.iloc[0:5, :]
print('\nTop 5 Crops According to Total Production:')
print(top5_crops)

# Top 5 crops by area of cultivation
crops_area = data[['Crop', 'Area']].groupby('Crop').sum()
ordered_crops_area = crops_area.sort_values(by='Area', ascending=False)
top5_crops_area = ordered_crops_area.iloc[0:5, :]
print('\nTop 5 Crops According to Total Area of Cultivation:')
print(top5_crops_area)

# Plotting pie charts for crops vs production and crops vs area
fig = plt.figure(figsize=(12, 20))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# Pie chart for crop vs area
top5_crops_area['Area'].plot(kind='pie', autopct='%1.1f%%', ax=ax1, fontsize=6.5)
ax1.set_ylabel('')
ax1.set_title('Top 5 Crops by Area')
ax1.legend(labels=top5_crops_area.index, loc='upper right', fontsize=6)

# Pie chart for crop vs production
top5_crops['Production'].plot(kind='pie', autopct='%1.1f%%', ax=ax2, fontsize=6.5)
ax2.set_ylabel('')
ax2.set_title('Top 5 Crops by Production')
ax2.legend(labels=top5_crops.index, loc='upper right', fontsize=6)

plt.show()

# Create new features if necessary
data['Production_per_Area'] = data['Production'] / data['Area']

# Prepare data for modeling
X = data[['Area', 'Crop_Year', 'State_Name', 'District_Name', 'Season', 'Crop']]
y = data['Production']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Visualization of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Production')
plt.ylabel('Predicted Production')
plt.title('Actual vs Predicted Production')
plt.show()

# Save the final dataset
data.to_csv('processed_crop_production_data.csv', index=False)
