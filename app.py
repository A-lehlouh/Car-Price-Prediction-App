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

# Load and preprocess dataset (cached)
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(r"C:\Users\3bdul\OneDrive\Desktop\car-price-prediction\car_price_prediction (Autosaved).csv")
    df.drop(['ID', 'Levy', 'Model'], axis=1, inplace=True)
    df['Engine volume'] = df['Engine volume'].str.split().str[0].astype(float)
    df['Mileage'] = df['Mileage'].str.split().str[0].astype(float)
    df = df[(df['Engine volume'] < 10) & (df['Price'] > 1000) & (df['Price'] < 200000)]
    return df

df = load_and_preprocess_data()

# Page setup
st.set_page_config(layout="wide")
st.title("🚗 Car Price Prediction App")

# User input

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

# Prepare user input for prediction
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

# Prediction
if st.button("🔮 Predict Price"):
    prediction = model.predict(user_input_encoded)[0]
    st.success(f"💰 Estimated Price: {int(prediction):,} USD")

# Cached functions for visualizations
@st.cache_data
def get_top_manufacturers(data):
    return data['Manufacturer'].value_counts().nlargest(10)

@st.cache_data
def get_avg_price_by_year(data):
    return data.groupby('Prod. year')['Price'].mean().reset_index()

@st.cache_data
def get_mileage_vs_price_sample(data):
    return data.sample(300, random_state=42)

# Visualizations
st.markdown("## 📊 Car Market Visual Analysis")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**Top Manufacturers**")
    top_manu = get_top_manufacturers(df)
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    sns.barplot(x=top_manu.values, y=top_manu.index, ax=ax1)
    ax1.set_xlabel("Count")
    ax1.set_ylabel("Manufacturer")
    fig1.tight_layout()
    st.pyplot(fig1)

with col_b:
    st.markdown("**Average Price by Year**")
    avg_price_by_year = get_avg_price_by_year(df)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.lineplot(data=avg_price_by_year, x='Prod. year', y='Price', ax=ax2, marker='o')
    ax2.set_xlabel("Production Year")
    ax2.set_ylabel("Average Price (USD)")
    fig2.tight_layout()
    st.pyplot(fig2)

with col_c:
    st.markdown("**Mileage vs Price**")
    sample_df = get_mileage_vs_price_sample(df)
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    sns.scatterplot(data=sample_df, x='Mileage', y='Price', alpha=0.5, ax=ax3)
    ax3.set_xlabel("Mileage (km)")
    ax3.set_ylabel("Price (USD)")
    ax3.set_xlim(0, 300000)  
    fig3.tight_layout()
    st.pyplot(fig3)
