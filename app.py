import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Streamlit page setup
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction App")
st.markdown("Predict whether a customer is likely to churn based on their telecom usage and demographic data.")

with st.form("churn_form"):
    st.subheader("Enter Customer Details")

    monthly_charge = st.number_input("Monthly Charge", min_value=0.0, value=50.0)
    city = st.selectbox("City", ["Los Angeles", "New York", "San Francisco", "Chicago", "Houston"])  # add more based on training data
    longitude = st.number_input("Longitude", value=-118.24)
    latitude = st.number_input("Latitude", value=34.05)
    zip_code = st.number_input("Zip Code", min_value=10000, max_value=99999, value=90001)
    number_of_referrals = st.slider("Number of Referrals", 0, 10, 2)
    age = st.slider("Age", 18, 90, 35)
    tenure_in_months = st.slider("Tenure in Months", 1, 100, 24)

    total_long_distance_charges = st.number_input("Total Long Distance Charges", value=30.0)
    total_revenue = st.number_input("Total Revenue", value=500.0)
    avg_monthly_long_distance_charges = st.number_input("Avg Monthly Long Distance Charges", value=10.0)
    total_charges = st.number_input("Total Charges", value=600.0)
    avg_monthly_gb_download = st.number_input("Avg Monthly GB Download", value=20.0)

    contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    number_of_dependents = st.slider("Number of Dependents", 0, 5, 1)
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"])
    engagement_score = st.slider("Engagement Score", 0.0, 10.0, 5.0)
    num_addon_services = st.slider("Number of Add-on Services", 0, 10, 2)
    offer = st.selectbox("Current Offer", ["None", "Offer A", "Offer B", "Offer C"])
    total_extra_data_charges = st.number_input("Total Extra Data Charges", value=40.0)
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    total_refunds = st.number_input("Total Refunds", value=10.0)
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Churn")

# On submit
if submitted:
    with st.spinner("Predicting..."):
        input_data = CustomData(
            monthly_charge=monthly_charge,
            city=city,
            longitude=longitude,
            latitude=latitude,
            zip_code=zip_code,
            number_of_referrals=number_of_referrals,
            age=age,
            tenure_in_months=tenure_in_months,
            total_long_distance_charges=total_long_distance_charges,
            total_revenue=total_revenue,
            avg_monthly_long_distance_charges=avg_monthly_long_distance_charges,
            total_charges=total_charges,
            avg_monthly_gb_download=avg_monthly_gb_download,
            contract=contract,
            number_of_dependents=number_of_dependents,
            payment_method=payment_method,
            engagement_score=engagement_score,
            num_addon_services=num_addon_services,
            offer=offer,
            total_extra_data_charges=total_extra_data_charges,
            paperless_billing=paperless_billing,
            total_refunds=total_refunds,
            multiple_lines=multiple_lines
        )

        df = input_data.get_data_as_data_frame()
        st.write("📄 Input Data Preview", df)

        predictor = PredictPipeline()
        result = predictor.predict(df)

        prediction = "Churn" if result[0] == 1 else "Not Churn"
        st.success(f"✅ The predicted customer status is: **{prediction}**")
