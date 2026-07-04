import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from xgboost import XGBClassifier
import streamlit as st
# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FraudGuard | AI Transaction Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .stButton>button {
        background-color: #004b87;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        width: 100%;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #003366;
        color: white;
    }
    h1, h2, h3 {
        color: #004b87;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        # 1. Initialize an empty XGBoost model
        model = XGBClassifier()
        # 2. Load the weights from the JSON file
        model.load_model("xgboost_fraud_model.json")
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# --- HEADER SECTION ---
st.title("🛡️ FraudGuard | Transaction Analysis System")
st.markdown("Enter transaction details below to run our AI-powered real-time fraud detection engine.")
st.markdown("---")

# --- MOCK ENCODERS (Update with your actual LabelEncoders) ---
# Because the notebook didn't save the LabelEncoders, we use mock dictionaries here.
# For a production app, load your saved .pkl encoders and use `encoder.transform()`.
GENDER_MAP = {"Male": 0, "Female": 1}
CATEGORY_MAP = {"Grocery": 0, "Entertainment": 1, "Healthcare": 2, "Gas/Transport": 3, "Online Retail": 4}
JOB_MAP = {"Engineer": 0, "Teacher": 1, "Doctor": 2, "Sales": 3, "Other": 4}
MERCHANT_MAP = {"Merchant A": 0, "Merchant B": 1, "Merchant C": 2, "Other": 3}

# --- SIDEBAR INPUTS ---
st.sidebar.header("📋 Transaction Input Form")

with st.sidebar.form("transaction_form"):
    st.subheader("Customer Details")
    gender = st.selectbox("Gender", options=list(GENDER_MAP.keys()))
    age = st.number_input("Age", min_value=18, max_value=120, value=35)
    job = st.selectbox("Job Title", options=list(JOB_MAP.keys()))
    city_pop = st.number_input("City Population", min_value=0, value=50000, step=1000)
    
    st.subheader("Transaction Details")
    amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0, step=10.0)
    category = st.selectbox("Category", options=list(CATEGORY_MAP.keys()))
    merchant = st.selectbox("Merchant", options=list(MERCHANT_MAP.keys()))
    
    st.subheader("Time & Location")
    # Date and time inputs to extract hour, day, month naturally
    trans_date = st.date_input("Transaction Date", datetime.date(2026, 7, 4))
    trans_time = st.time_input("Transaction Time", datetime.time(12, 30))
    
    # Location coordinates
    col_lat, col_long = st.columns(2)
    lat = col_lat.number_input("Cust Latitude", value=40.7128, format="%.4f")
    long = col_long.number_input("Cust Longitude", value=-74.0060, format="%.4f")
    
    col_mlat, col_mlong = st.columns(2)
    merch_lat = col_mlat.number_input("Merch Latitude", value=40.7306, format="%.4f")
    merch_long = col_mlong.number_input("Merch Longitude", value=-73.9866, format="%.4f")

    # Generate Unix timestamp dynamically
    dt_combined = datetime.datetime.combine(trans_date, trans_time)
    unix_time = int(dt_combined.timestamp())

    submit_button = st.form_submit_button(label="🔍 Analyze Transaction")

# --- MAIN DASHBOARD AREA ---
if submit_button:
    if model is None:
        st.error("Cannot run analysis without the model.")
    else:
        # 1. Prepare data matching exact column order from the Jupyter Notebook
        input_data = pd.DataFrame([[
            MERCHANT_MAP[merchant],
            CATEGORY_MAP[category],
            amt,
            GENDER_MAP[gender],
            lat,
            long,
            city_pop,
            JOB_MAP[job],
            unix_time,
            merch_lat,
            merch_long,
            trans_time.hour,
            trans_date.day,
            trans_date.month,
            age
        ]], columns=[
            'merchant', 'category', 'amt', 'gender', 'lat', 'long', 'city_pop', 
            'job', 'unix_time', 'merch_lat', 'merch_long', 'hour', 'day', 'month', 'age'
        ])
        
        # 2. Make Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] # Probability of fraud (Class 1)
        
        # 3. Display Results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
            if prediction == 1:
                st.error("### 🚨 FRAUD DETECTED")
                st.write("This transaction exhibits patterns consistent with fraudulent activity. Recommend immediate block and customer verification.")
            else:
                st.success("### ✅ TRANSACTION APPROVED")
                st.write("This transaction appears legitimate based on current historical models.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.subheader("Risk Analysis")
            risk_color = "red" if prediction == 1 else "green"
            
            # Display metrics
            st.metric("Fraud Probability", f"{probability * 100:.2f}%", 
                      delta="High Risk" if prediction == 1 else "Low Risk", 
                      delta_color="inverse")
            
            st.progress(float(probability))
            
        # Optional: Display the raw feature vector for transparency/debugging
        with st.expander("View Raw Feature Vector"):
            st.dataframe(input_data)
else:
    # Default landing screen
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", "99.77%")
    col2.metric("ROC-AUC Score", "96.17%")
    col3.metric("Status", "System Online")
    
    st.info("👈 Enter the transaction details in the sidebar and click 'Analyze Transaction' to see the model's prediction.")