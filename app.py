import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model pipeline
with open('pipe.pkl', 'rb') as f:
    pipe = pickle.load(f)

# Title
st.title("💻 Laptop Price Predictor (GBP)")

# Dropdown inputs
data = pd.read_csv("traineddata.csv")  # for dropdown unique values

company = st.selectbox('Brand', data['Company'].unique())
type_ = st.selectbox('Type', data['TypeName'].unique())
ram = st.selectbox('Ram (in GB)', sorted(data['Ram'].unique()))
os = st.selectbox('OS', data['OpSys'].unique())
weight = st.number_input('Weight of the laptop (kg)', min_value=0.1, step=0.1)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.number_input('Screen Size (inches)', min_value=0.0, step=0.1)
resolution = st.text_input('Screen Resolution (e.g., 1920x1080)', '1920x1080')
cpu = st.selectbox('CPU Brand', data['CPU_name'].unique())
hdd = st.selectbox('HDD (in GB)', sorted(data['HDD'].unique()))
ssd = st.selectbox('SSD (in GB)', sorted(data['SSD'].unique()))
gpu = st.selectbox('GPU Brand', data['Gpu brand'].unique())

if st.button('Predict Price'):
    # Convert categorical Yes/No
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    try:
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size
    except:
        st.error("❌ Please enter resolution in correct format (e.g. 1920x1080)")
        st.stop()

    # Build input DataFrame
    input_dict = {
        'Company': company,
        'TypeName': type_,
        'Ram': ram,
        'OpSys': os,
        'Weight': weight,
        'TouchScreen': touchscreen,
        'IPS': ips,
        'PPI': ppi,
        'CPU_name': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu
    }
    input_df = pd.DataFrame([input_dict])

    # Predict and convert from log price
    price_in_inr = np.exp(pipe.predict(input_df)[0])
    price_in_gbp = price_in_inr / 105  # Adjust conversion rate as needed

    st.success(f"💷 Predicted Price: £{price_in_gbp:,.2f}")