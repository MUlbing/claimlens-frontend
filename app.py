import streamlit as st
import requests

"""
# Claimlens
"""
url = st.secrets['API_URL']

#st.title("Predict the refund intent of a customer")

#st.markdown(
#   """
#Please enter the customer message and click the **Predict** button to see the predicted refund intent.
#"""
#)
st.set_page_config(
    page_title="ClaimLens - Intelligent recognition of customer refund intent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search-property.png", width=64)
    st.title("ClaimLens Console")
    st.markdown("Refund intent recognition based on NLP pipeline processing")

    st.markdown("---")
    st.subheader("Model Selection")
    st.info("Current Model: **LinearSVC / ComplementNB + TF-IDF (1-2 gram)**")

    threshold = st.slider("Intent Decision Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    st.markdown("---")
    st.caption("© 2026 ClaimLens Team. All rights reserved.")

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
            intent = prediction['Refund Intent']
            prediction_prob = prediction_prob['Refund Intent Probability']

            st.success(f"Predicted Refund Intent: **{intent}**")
            st.success(f"Predicted Refund Probability: **{prediction_prob}**")

        except requests.exceptions.RequestException as e:
            st.error(f"API Request Failed: {e}")
