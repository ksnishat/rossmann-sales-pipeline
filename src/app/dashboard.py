import streamlit as st
import requests
import os  # NEW: To read environment variables

# 1. Define API URL (Dynamic)
# Docker will send "http://api:8000/predict". Localhost will use the default.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# 2. Page Setup
st.set_page_config(page_title="Rossmann Sales AI", layout="centered")
st.title("Rossmann Sales Prediction")
st.write("Enter store details below to predict daily sales.")

# 3. Create Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        store_id = st.number_input("Store ID", min_value=1, value=1115)
        day_of_week = st.slider("Day of Week (1=Mon, 7=Sun)", 1, 7, 5)
        promo = st.selectbox("Is there a Promo?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        school_holiday = st.selectbox("School Holiday?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

    with col2:
        store_type = st.selectbox("Store Type", ["a", "b", "c", "d"])
        assortment = st.selectbox("Assortment Level", ["a", "b", "c"])
        distance = st.number_input("Competitor Distance (m)", value=500.0)
        promo2 = st.selectbox("Promo 2 Active?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    
    # Submit Button
    submitted = st.form_submit_button("🔮 Predict Sales")

# 4. Handle Submission
if submitted:
    # Prepare the JSON payload
    payload = {
        "Store": store_id,
        "DayOfWeek": day_of_week,
        "Promo": promo,
        "SchoolHoliday": school_holiday,
        "StoreType": store_type,
        "Assortment": assortment,
        "CompetitionDistance": distance,
        "Promo2": promo2
    }
    
    # Send to API
    try:
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            prediction = response.json()["predicted_sales"]
            st.success(f"Predicted Sales: €{prediction:,.2f}")
        else:
            st.error(f"Error: {response.text}")
            
    except Exception as e:
        st.error(f"Failed to connect to API. Is it running? Error: {e}")