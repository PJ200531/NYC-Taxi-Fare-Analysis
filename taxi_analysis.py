# 1️⃣ Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from geopy.distance import geodesic

# 2️⃣ Load dataset
df = pd.read_csv('train.csv')  # download from Kaggle: train.csv

# Quick look
print(df.head())
print(df.info())

# 3️⃣ Basic Cleaning
# Drop rows with missing fare amounts
df = df.dropna(subset=['fare_amount', 'pickup_longitude', 'pickup_latitude',
                       'dropoff_longitude', 'dropoff_latitude'])

# Remove outliers (fare <= 0 or impossible coordinates)
df = df[df['fare_amount'] > 0]
df = df[(df['pickup_latitude'].between(40, 42)) & (df['dropoff_latitude'].between(40, 42))]
df = df[(df['pickup_longitude'].between(-75, -73)) & (df['dropoff_longitude'].between(-75, -73))]

# 4️⃣ Feature Engineering
# Haversine distance function
def haversine_distance(row):
    pickup = (row['pickup_latitude'], row['pickup_longitude'])
    dropoff = (row['dropoff_latitude'], row['dropoff_longitude'])
    return geodesic(pickup, dropoff).km

df['distance_km'] = df.apply(haversine_distance, axis=1)

# Extract time features
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['month'] = df['pickup_datetime'].dt.month

# 5️⃣ Exploratory Data Analysis (quick)
plt.figure(figsize=(8,5))
sns.histplot(df['fare_amount'], bins=50, kde=True, color='teal')
plt.title("Fare Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x='distance_km', y='fare_amount', data=df, alpha=0.3)
plt.title("Fare vs Distance")
plt.show()

# 6️⃣ Modeling
# Features and target
X = df[['distance_km', 'passenger_count', 'hour', 'day_of_week', 'month']]
y = df['fare_amount']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Choose model
model = RandomForestRegressor(n_estimators=100, random_state=42)
# OR LinearRegression()
# model = LinearRegression()

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 7️⃣ Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Feature Importance (for RandomForest)
if isinstance(model, RandomForestRegressor):
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(6,4))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='magma')
    plt.title("Feature Importance")
    plt.show()

new_trip = pd.DataFrame({
    'distance_km': [5.2],
    'passenger_count': [2],
    'hour': [18],
    'day_of_week': [4],  # Friday
    'month': [10]
})

# Scale features
new_trip_scaled = scaler.transform(new_trip)

# Predict fare
predicted_fare = model.predict(new_trip_scaled)
print(f"Predicted Fare: ${predicted_fare[0]:.2f}")
