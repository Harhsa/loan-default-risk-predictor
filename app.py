import streamlit as st
import pandas as pd
import joblib
import numpy as np

from src.features import create_features


# -------------------------------
# Load model & feature schema
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/rf_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, feature_names


model, feature_names = load_artifacts()
st.divider()
st.header("👤 Project Credits")

st.markdown("""
**Developed by:** Harsha Vardhan Dasari  
**Role:** Data Scientist / Machine Learning Engineer  

📧 **Email:** dasariharshavardhan2002@gmail.com  
🔗 **LinkedIn:** https://www.linkedin.com/in/harsha-vardhan-dasari-no1/  
💻 **GitHub:** https://github.com/Harhsa  

This project demonstrates the end-to-end development of a real-world
Machine Learning system, including data preprocessing, feature engineering,
model training, evaluation, and deployment as an interactive web application.
""")

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Loan Default Risk Predictor", layout="centered")

st.title("💳 Loan Default Risk Prediction")
st.write(
    "This application predicts the probability that a credit card customer will default "
    "based on historical repayment and billing behavior."
)

st.divider()

st.subheader("Enter Customer Details")

LIMIT_BAL = st.number_input("Credit Limit", min_value=1000, step=1000, value=50000)
AGE = st.number_input("Age", min_value=18, max_value=100, value=30)

SEX = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
EDUCATION = st.selectbox("Education Level", options=[1, 2, 3, 4])
MARRIAGE = st.selectbox("Marital Status", options=[1, 2, 3])

st.markdown("### Repayment Status (Last 6 Months)")
PAY_0 = st.number_input("Last Month", value=0)
PAY_2 = st.number_input("2 Months Ago", value=0)
PAY_3 = st.number_input("3 Months Ago", value=0)
PAY_4 = st.number_input("4 Months Ago", value=0)
PAY_5 = st.number_input("5 Months Ago", value=0)
PAY_6 = st.number_input("6 Months Ago", value=0)

st.markdown("### Bill Amounts (Last 6 Months)")
BILL_AMT1 = st.number_input("Bill Month 1", value=2000)
BILL_AMT2 = st.number_input("Bill Month 2", value=2000)
BILL_AMT3 = st.number_input("Bill Month 3", value=2000)
BILL_AMT4 = st.number_input("Bill Month 4", value=2000)
BILL_AMT5 = st.number_input("Bill Month 5", value=2000)
BILL_AMT6 = st.number_input("Bill Month 6", value=2000)

st.markdown("### Payment Amounts (Last 6 Months)")
PAY_AMT1 = st.number_input("Payment Month 1", value=1000)
PAY_AMT2 = st.number_input("Payment Month 2", value=1000)
PAY_AMT3 = st.number_input("Payment Month 3", value=1000)
PAY_AMT4 = st.number_input("Payment Month 4", value=1000)
PAY_AMT5 = st.number_input("Payment Month 5", value=1000)
PAY_AMT6 = st.number_input("Payment Month 6", value=1000)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Default Risk"):
    input_data = pd.DataFrame([{
        "LIMIT_BAL": LIMIT_BAL,
        "AGE": AGE,
        "PAY_0": PAY_0,
        "PAY_2": PAY_2,
        "PAY_3": PAY_3,
        "PAY_4": PAY_4,
        "PAY_5": PAY_5,
        "PAY_6": PAY_6,
        "BILL_AMT1": BILL_AMT1,
        "BILL_AMT2": BILL_AMT2,
        "BILL_AMT3": BILL_AMT3,
        "BILL_AMT4": BILL_AMT4,
        "BILL_AMT5": BILL_AMT5,
        "BILL_AMT6": BILL_AMT6,
        "PAY_AMT1": PAY_AMT1,
        "PAY_AMT2": PAY_AMT2,
        "PAY_AMT3": PAY_AMT3,
        "PAY_AMT4": PAY_AMT4,
        "PAY_AMT5": PAY_AMT5,
        "PAY_AMT6": PAY_AMT6,
        "SEX": SEX,
        "EDUCATION": EDUCATION,
        "MARRIAGE": MARRIAGE,
    }])

    # Feature engineering
    input_data = create_features(input_data)

    # One-hot encoding
    input_data = pd.get_dummies(
        input_data,
        columns=["SEX", "EDUCATION", "MARRIAGE"],
        drop_first=True
    )

    # Align features
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # Predict
    prediction_proba = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    st.metric(
        label="Probability of Default",
        value=f"{prediction_proba * 100:.2f}%"
    )

    if prediction_proba >= 0.6:
        st.error("High Risk of Default")
    elif prediction_proba >= 0.3:
        st.warning("Medium Risk of Default")
    else:
        st.success("Low Risk of Default")
st.divider()
st.header("📌 About This Project")

st.markdown("""
### Loan Default Risk Prediction System

This project is an **end-to-end Machine Learning application** built to predict the probability of 
a customer defaulting on credit card payments.

It demonstrates how Data Science models are:
- designed,
- trained,
- evaluated,
- and deployed
in real-world financial risk systems.
""")
st.subheader("🔍 How It Works")

st.markdown("""
**Step 1: Data Collection**  
Historical credit card customer data including repayment history, billing amounts, and demographics.

**Step 2: Feature Engineering**  
Key risk indicators are created such as:
- Average billing amount  
- Average payment amount  
- Credit utilization ratio  

**Step 3: Model Training**  
A Random Forest classifier is trained to learn non-linear risk patterns while handling class imbalance.

**Step 4: Model Evaluation**  
The model is evaluated using ROC-AUC to ensure strong discrimination between defaulters and non-defaulters.

**Step 5: Real-Time Prediction**  
User inputs are transformed using the same feature pipeline and passed to the trained model to generate risk probability.
""")
