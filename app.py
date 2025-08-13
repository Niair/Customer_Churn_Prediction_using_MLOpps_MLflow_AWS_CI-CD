import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import os

# Streamlit page setup
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

os.environ["STREAMLIT_SERVER_PORT"] = "7860"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

st.title("📉 Customer Churn Prediction App")
st.markdown("Predict whether a customer is likely to churn based on their telecom usage and demographic data.")

with st.form("churn_form"):
    st.subheader("Enter Customer Details")
    
    # Personal Information
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 90, 35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        married = st.selectbox("Married", ["Yes", "No"])
        number_of_dependents = st.slider("Number of Dependents", 0, 5, 1)
        zip_code = st.number_input("Zip Code", min_value=10000, max_value=99999, value=90001)
        
    # Location Information
    with col2:
        city = st.selectbox("City", ["Los Angeles", "New York", "San Francisco", "Chicago", "Houston"])
        longitude = st.number_input("Longitude", value=-118.24)
        latitude = st.number_input("Latitude", value=34.05)
        internet_type = st.selectbox("Internet Type", ["Cable", "DSL", "Fiber Optic", "None"])
        
    # Account Information
    st.subheader("Account & Billing Details")
    col1, col2 = st.columns(2)
    with col1:
        tenure_in_months = st.slider("Tenure in Months", 1, 100, 24)
        contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
        payment_method = st.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        
    with col2:
        monthly_charge = st.number_input("Monthly Charge", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges", value=600.0)
        total_revenue = st.number_input("Total Revenue", value=500.0)
        offer = st.selectbox("Current Offer", ["None", "Offer A", "Offer B", "Offer C"])
        
    # Usage Details
    st.subheader("Usage Statistics")
    col1, col2 = st.columns(2)
    with col1:
        number_of_referrals = st.slider("Number of Referrals", 0, 10, 2)
        engagement_score = st.slider("Engagement Score", 0.0, 10.0, 5.0)
        avg_monthly_gb_download = st.number_input("Avg Monthly GB Download", value=20.0)
        num_addon_services = st.slider("Number of Add-on Services", 0, 10, 2)
        
    with col2:
        total_long_distance_charges = st.number_input("Total Long Distance Charges", value=30.0)
        avg_monthly_long_distance_charges = st.number_input("Avg Monthly Long Distance Charges", value=10.0)

    submitted = st.form_submit_button("Predict Churn")

# On submit
if submitted:
    with st.spinner("Predicting..."):
        # ONLY PASS PARAMETERS DEFINED IN CustomData CLASS
        input_data = CustomData(
            monthly_charge=monthly_charge,
            zip_code=zip_code,
            longitude=longitude,
            age=age,
            latitude=latitude,
            total_long_distance_charges=total_long_distance_charges,
            tenure_in_months=tenure_in_months,
            total_revenue=total_revenue,
            number_of_referrals=number_of_referrals,
            total_charges=total_charges,
            avg_monthly_long_distance_charges=avg_monthly_long_distance_charges,
            avg_monthly_gb_download=avg_monthly_gb_download,
            number_of_dependents=number_of_dependents,
            engagement_score=engagement_score,
            num_addon_services=num_addon_services,
            city=city,
            contract=contract,
            payment_method=payment_method,
            offer=offer,
            paperless_billing=paperless_billing,
            gender=gender,
            married=married,
            internet_type=internet_type
        )

        df = input_data.get_data_as_data_frame()
        st.write("📄 Input Data Preview", df)

        predictor = PredictPipeline()
        result = predictor.predict(df)

        prediction = "Churn" if result[0] == 1 else "Not Churn"
        st.success(f"✅ The predicted customer status is: **{prediction}**")