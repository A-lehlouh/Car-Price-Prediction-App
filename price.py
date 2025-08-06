import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load and clean dataset
df = pd.read_csv(r"C:\Users\3bdul\OneDrive\Desktop\car-price-prediction\car_price_prediction (Autosaved).csv")
df.drop(['ID', 'Levy', 'Model'], axis=1, inplace=True)
df['Engine volume'] = df['Engine volume'].str.split().str[0].astype(float)
df['Mileage'] = df['Mileage'].str.split().str[0].astype(float)
df = df[(df['Engine volume'] < 10) & (df['Price'] > 1000) & (df['Price'] < 200000)]

# Prepare data for model
df_encoded = pd.get_dummies(df.drop('Price', axis=1), drop_first=True)
X = df_encoded
y = df['Price']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Layout settings
st.set_page_config(layout="wide")
st.title("🚗 Car Price Prediction App")
st.markdown("Enter your car's information below to estimate its market value:")

# Input columns
col1, col2, col3 = st.columns(3)
with col1:
    manufacturer = st.selectbox("Manufacturer", sorted(df['Manufacturer'].unique()))
    engine_volume = st.number_input("Engine Volume", min_value=0.5, max_value=10.0, value=2.0, step=0.1)

with col2:
    prod_year = st.number_input("Production Year", min_value=1990, max_value=2025, value=2015)
    cylinders = st.number_input("Cylinders", min_value=2, max_value=16, value=4)

with col3:
    fuel_type = st.selectbox("Fuel Type", sorted(df['Fuel type'].unique()))
    mileage = st.number_input("Mileage (in km)", min_value=0, max_value=1000000, value=100000, step=1000)

# Prepare user input
user_data = pd.DataFrame({
    'Prod. year': [prod_year],
    'Engine volume': [engine_volume],
    'Cylinders': [cylinders],
    'Mileage': [mileage],
    'Manufacturer': [manufacturer],
    'Fuel type': [fuel_type]
})
user_data_encoded = pd.get_dummies(user_data)
user_data_encoded = user_data_encoded.reindex(columns=X.columns, fill_value=0)

# Prediction
if st.button("🔮 Predict Price"):
    predicted_price = model.predict(user_data_encoded)[0]
    st.success(f"💰 Estimated Car Price: {int(predicted_price):,} USD")

# --- 📊 Visual Analysis ---

st.markdown("## 📊 Car Market Visual Analysis")

col_a, col_b, col_c = st.columns(3)

# Chart 1: Bar - Top Manufacturers
with col_a:
    st.markdown("**Top Manufacturers**")
    top_manu = df['Manufacturer'].value_counts().nlargest(10)
    fig1, ax1 = plt.subplots()
    sns.barplot(x=top_manu.values, y=top_manu.index, ax=ax1, palette="Blues_d")
    ax1.set_xlabel("Count")
    st.pyplot(fig1)

# Chart 2: Line - Avg Price by Year
with col_b:
    st.markdown("**Average Price by Year**")
    avg_price_by_year = df.groupby('Prod. year')['Price'].mean().reset_index()
    fig2, ax2 = plt.subplots()
    sns.lineplot(data=avg_price_by_year, x='Prod. year', y='Price', marker='o', ax=ax2)
    ax2.set_ylabel("Avg Price")
    st.pyplot(fig2)

# Chart 3: Scatter - Mileage vs Price
with col_c:
    st.markdown("**Mileage vs Price**")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df.sample(300), x='Mileage', y='Price', alpha=0.5, ax=ax3)
    st.pyplot(fig3)
