import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("final_model.pkl")

# Sidebar info
st.sidebar.title("🏦 Loan Eligibility Predictor")
st.sidebar.info(
    """
    This app predicts whether a loan will be approved or not.
    Fill in the applicant's details and click Predict.
    """
)

st.title("🏦 Loan Eligibility Prediction App")
st.write("Enter applicant details below to check loan approval status.")

# --- User Input ---
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

# --- Preprocess Input ---
def preprocess_input():
    data = {
        "Gender": 1 if gender=="Male" else 0,
        "Married": 1 if married=="Yes" else 0,
        "Education": 1 if education=="Graduate" else 0,
        "Self_Employed": 1 if self_employed=="Yes" else 0,
        "ApplicantIncome": applicant_income,
        
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Dependents_1": 1 if dependents=="1" else 0,
        "Dependents_2": 1 if dependents=="2" else 0,
        "Dependents_3+": 1 if dependents=="3+" else 0,
        "Property_Area_Semiurban": 1 if property_area=="Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area=="Urban" else 0
    }
    return pd.DataFrame([data])

# --- Prediction ---
if st.button("Predict Loan Eligibility"):
    input_df = preprocess_input()
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    # Result text with color
    if prediction == 1:
        st.success(f"✅ Loan Approved! (Probability: {probability[1]*100:.2f}%)")
    else:
        st.error(f"❌ Loan Not Approved (Probability: {probability[0]*100:.2f}%)")

    # Plot probability as bar chart
    st.subheader("Prediction Probability")
    prob_df = pd.DataFrame({
        "Status": ["Not Approved", "Approved"],
        "Probability": [probability[0], probability[1]]
    })

    fig, ax = plt.subplots()
    ax.bar(prob_df["Status"], prob_df["Probability"], color=["red", "green"])
    ax.set_ylim([0,1])
    ax.set_ylabel("Probability")
    st.pyplot(fig)

