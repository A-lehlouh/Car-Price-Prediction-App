# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle
import os

# Load and clean dataset
df = pd.read_csv(r"C:\Users\3bdul\OneDrive\Desktop\car-price-prediction\car_price_prediction (Autosaved).csv")
df.drop(['ID', 'Levy', 'Model'], axis=1, inplace=True)
df['Engine volume'] = df['Engine volume'].str.split().str[0].astype(float)
df['Mileage'] = df['Mileage'].str.split().str[0].astype(float)
df = df[(df['Engine volume'] < 10) & (df['Price'] > 1000) & (df['Price'] < 200000)]

# Encode features
df_encoded = pd.get_dummies(df.drop('Price', axis=1), drop_first=True)
X = df_encoded
y = df['Price']

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and feature columns
with open("rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model and columns saved!")

