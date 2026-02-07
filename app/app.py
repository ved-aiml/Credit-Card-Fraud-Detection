import streamlit as st
import numpy as np
import joblib

model = joblib.load("app/model.pkl")

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Predict the probability of a transaction being fraudulent.")

st.markdown("---")
st.subheader("Transaction Input")

amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)

features = []
for i in range(1, 29):
    features.append(
        st.slider(f"V{i}", -10.0, 10.0, 0.0)
    )

input_data = np.array([[0] + features + [amount]])
if st.button("Predict Fraud Risk"):
    proba = model.predict_proba(input_data)[0][1]

    st.metric("Fraud Probability", f"{proba:.2%}")

    if proba < 0.3:
        st.success("✅ Transaction Allowed")
    elif proba < 0.7:
        st.warning("⚠️ Transaction Needs Manual Review")
    else:
        st.error("🚨 Transaction Blocked")

