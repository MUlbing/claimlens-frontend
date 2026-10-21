import os
import streamlit as st
import requests

st.set_page_config(
    page_title="ClaimLens - Two-Stage Refund Decision System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# One base URL for the API. Default is Cloud Run.
# To test against a local API: CLAIMLENS_API_URL=http://127.0.0.1:8000 uv run streamlit run app.py
API_URL = os.getenv(
    "CLAIMLENS_API_URL",
    "https://claimlens-api-392994371879.europe-west3.run.app",
)
URL_STAGE1 = f"{API_URL}/predict"
URL_STAGE2 = f"{API_URL}/more_context"

# Model 1 probability below this goes straight to an agent.
# The Yes/No label always comes from the API. This number only picks case 2a or 2b.
LOW_INTENT_CUTOFF = 0.25

# One sentence for every live handoff to an agent
AGENT_LINE = "An agent has been assigned to you and will be with you shortly."

# ----------------------------------------------------
# 1. Initialize Session State
# ----------------------------------------------------
if "first_message" not in st.session_state:
    st.session_state.first_message = ""
if "stage1_result" not in st.session_state:
    st.session_state.stage1_result = None
if "stage2_result" not in st.session_state:
    st.session_state.stage2_result = None
if "step" not in st.session_state:
    # input | context | choice | approved | agent | error
    st.session_state.step = "input"
if "agent_reason" not in st.session_state:
    # low_intent | customer_request | retry_no | model2_no
    st.session_state.agent_reason = None


def reset_flow():
    st.session_state.first_message = ""
    st.session_state.stage1_result = None
    st.session_state.stage2_result = None
    st.session_state.step = "input"
    st.session_state.agent_reason = None


# ----------------------------------------------------
# 2. Sidebar Configuration
# ----------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search-property.png", width=64)
    st.title("ClaimLens Console")
    st.markdown("Refund intent recognition based on NLP pipeline processing")

    st.markdown("---")
    st.subheader("Model Pipeline")
    st.info("**Stage 1**: Intent Detection\n\n**Stage 2**: Case Resolution & Reasoning")

    #threshold = st.slider("Intent Decision Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    if st.button("🔄 Reset Conversation", use_container_width=True):
        reset_flow()
        st.rerun()

    st.markdown("---")
    st.caption("© 2026 ClaimLens Team. All rights reserved.")

# ----------------------------------------------------
# 3. Stage 1: Initial Customer Message Input
# ----------------------------------------------------
# st.title("🔍 ClaimLens - Customer Service Intelligence")

st.markdown(
    """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="margin-bottom: 0.25rem;">🔍 ClaimLens</h1>
        <h3 style="color: #6c757d; font-weight: 400; margin-top: 0;">Intelligen Customer Service</h3>
    </div>
    """,
    unsafe_allow_html=True
)

with st.form("Stage 1 - Initial Message"):
    customer_message = st.text_area(
        "Please enter the customer message",
        value=st.session_state.first_message,
        placeholder="e.g. My package arrived wet and the leather jacket is destroyed. Can I get a refund?",
        height=120
    )
    submitted = st.form_submit_button("Predict Intent", use_container_width=True)

    if submitted:
        if not customer_message.strip():
            st.warning("Please enter a message first.")
        else:
            st.session_state.first_message = customer_message
            st.session_state.stage2_result = None

            params = {
                "customer_message": customer_message,
            }

            try:
                response = requests.get(URL_STAGE1, params=params, timeout=30)
                response.raise_for_status()
                result = response.json()

                intent = result.get("Refund Intent", "Yes" if result.get("refund_intent") == 1 else "No")
                prob = float(result.get("Refund Intent Probability", result.get("probability", 0.0)))

                #prediction = result.get("refund_intent", result)
                #prediction_prob = result.get("probability", result)
                #intent = prediction.get("Refund Intent", result.get("Refund Intent", "No"))
               # prob = prediction_prob.get("Refund Intent Probability", result.get("Refund Intent Probability", 0.0))

                st.session_state.stage1_result = {
                    "intent": intent,
                    "probability": prob
                }

            except requests.exceptions.RequestException as e:
                st.error(f"Stage 1 API Request Failed: {e}")

# Display Stage 1 Result
if st.session_state.stage1_result:
    res1 = st.session_state.stage1_result
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if res1["intent"] == "Yes":
            st.success(f"Predicted Claim intent: **{res1['intent']}**")
        else:
            st.info(f"Predicted Claim intent: **{res1['intent']}**")
    with col2:
        st.metric(label="Claim Intent Probability", value=f"{res1['probability']}")

# ----------------------------------------------------
# 4. Stage 2: Follow-up Input Form (Triggered when Intent is 'Yes')
# ----------------------------------------------------
if st.session_state.stage1_result and st.session_state.stage1_result["intent"] == "Yes":
    st.markdown("---")
    st.subheader("📋 Step 2: Additional Case Details")
    st.info("Refund intent confirmed. Please provide further details (e.g. order number, item condition, specific issue) to proceed with the resolution.")

    with st.form("Stage 2 - Follow-up Details"):
        additional_info = st.text_area(
            "Please enter additional customer information",
            placeholder="e.g. Order #12345. The zipper is completely broken and torn off.",
            height=100
        )
        submitted_stage2 = st.form_submit_button("Submit to Stage 2 Model", use_container_width=True)

        if submitted_stage2:
            if not additional_info.strip():
                st.warning("Please enter additional details.")
            else:
                combined_text = f"{st.session_state.first_message} {additional_info}"
                payload = {
                    "initial_message": st.session_state.first_message,
                    "additional_message": additional_info,
                    "combined_message": combined_text
                }

                try:
                    with st.spinner("Processing second model inference..."):
                        params = {"customer_message": combined_text}
                        response = requests.get(URL_STAGE2, params=params, timeout=30)
                        response.raise_for_status()
                        result = response.json()

                        intent = result.get("Money Back", "Yes" if result.get("refund_intent") == 1 else "No")
                        prob = float(result.get("Chance of money loss", result.get("probability", 0.0)))

                        st.session_state.stage2_result = {
                                            "Money back": intent,
                                            "Chance of money loss": prob
                                        }

                except requests.exceptions.RequestException as e:
                    st.error(f"Stage 2 API Request Failed: {e}")

# ----------------------------------------------------
# 5. Stage 2 Result Display
# ----------------------------------------------------
if st.session_state.stage2_result:
    st.markdown("---")
    st.subheader("🏁 Stage 2 Final Resolution")
    res2 = st.session_state.stage2_result

    st.json(res2)
