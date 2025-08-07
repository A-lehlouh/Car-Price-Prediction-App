import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and columns
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Load dataset for UI and visualizations
df = pd.read_csv(r"C:\Users\3bdul\OneDrive\Desktop\car-price-prediction\car_price_prediction (Autosaved).csv")
df.drop(['ID', 'Levy', 'Model'], axis=1, inplace=True)
df['Engine volume'] = df['Engine volume'].str.split().str[0].astype(float)
df['Mileage'] = df['Mileage'].str.split().str[0].astype(float)
df = df[(df['Engine volume'] < 10) & (df['Price'] > 1000) & (df['Price'] < 200000)]

# Layout settings
st.set_page_config(layout="wide")
st.title("🚗 Car Price Prediction App")

# Input
st.markdown("Enter your car's information below:")

col1, col2, col3 = st.columns(3)
with col1:
    manufacturer = st.selectbox("Manufacturer", sorted(df['Manufacturer'].unique()))
    engine_volume = st.number_input("Engine Volume", 0.5, 10.0, 2.0, 0.1)

with col2:
    prod_year = st.number_input("Production Year", 1990, 2025, 2015)
    cylinders = st.number_input("Cylinders", 2, 16, 4)

with col3:
    fuel_type = st.selectbox("Fuel Type", sorted(df['Fuel type'].unique()))
    mileage = st.number_input("Mileage (km)", 0, 1000000, 100000, 1000)

# Prepare [input]
user_input = pd.DataFrame({
    'Prod. year': [prod_year],
    'Engine volume': [engine_volume],
    'Cylinders': [cylinders],
    'Mileage': [mileage],
    'Manufacturer': [manufacturer],
    'Fuel type': [fuel_type]
})

user_input_encoded = pd.get_dummies(user_input)
user_input_encoded = user_input_encoded.reindex(columns=model_columns, fill_value=0)

if st.button("🔮 Predict Price"):
    prediction = model.predict(user_input_encoded)[0]
    st.success(f"💰 Estimated Price: {int(prediction):,} USD")

# 📊 Visualizations
st.markdown("## 📊 Car Market Visual Analysis")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**Top Manufacturers**")
    top_manu = df['Manufacturer'].value_counts().nlargest(10)
    fig1, ax1 = plt.subplots()
    sns.barplot(x=top_manu.values, y=top_manu.index, ax=ax1)
    st.pyplot(fig1)

with col_b:
    st.markdown("**Average Price by Year**")
    avg_price_by_year = df.groupby('Prod. year')['Price'].mean().reset_index()
    fig2, ax2 = plt.subplots()
    sns.lineplot(data=avg_price_by_year, x='Prod. year', y='Price', ax=ax2, marker='o')
    st.pyplot(fig2)

with col_c:
    st.markdown("**Mileage vs Price**")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df.sample(300), x='Mileage', y='Price', alpha=0.5, ax=ax3)
    st.pyplot(fig3)
