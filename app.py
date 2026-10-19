import streamlit as st
import requests

"""
# Zoologist front
"""
url = st.secrets['API_URL']

st.title("Claimlens - Predict the refund intent of a customer")

st.markdown(
    """
Please enter the customer message and click the **Predict** button to see the predicted refund intent.
"""
)

with st.form("Customer message"):

    customer_message = st.text_area("Please enter the customer message")

    submitted = st.form_submit_button("Predict")

    if submitted:
        params = {
            "customer_message": customer_message
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            prediction = result.get("refund_intent", result)
            prediction_prob = result.get("probability", result)
            intent = "Yes" if prediction == 1 else "No"

            st.success(f"Predicted Refund Intent: **{intent}**")
            st.success(f"Predicted Refund Probability: **{prediction_prob}**")

        except requests.exceptions.RequestException as e:
            st.error(f"API Request Failed: {e}")
