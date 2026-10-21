import streamlit as st
import requests

st.set_page_config(
    page_title="ClaimLens - Two-Stage Refund Decision System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


URL_STAGE1 = "https://claimlens-api-392994371879.europe-west3.run.app/predict"
URL_STAGE2 = "https://claimlens-api-392994371879.europe-west3.run.app/more_context"


if "first_message" not in st.session_state:
    st.session_state.first_message = ""
if "stage1_result" not in st.session_state:
    st.session_state.stage1_result = None
if "stage2_result" not in st.session_state:
    st.session_state.stage2_result = None


def reset_flow():
    st.session_state.first_message = ""
    st.session_state.stage1_result = None
    st.session_state.stage2_result = None


with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search-property.png", width=64)
    st.title("ClaimLens Console")
    st.markdown("Refund intent recognition based on NLP pipeline processing")

    st.markdown("---")
    st.subheader("Model Pipeline")
    st.info("**Stage 1**: Claim Intent Detection \n\n**Stage 2**: Refund Decision with Additional Context")

    if st.button("🔄 Reset Conversation", use_container_width=True):
        reset_flow()
        st.rerun()

    st.markdown("---")
    st.caption("© 2026 ClaimLens Team. All rights reserved.")

st.markdown(
    """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="margin-bottom: 0.25rem;">🔍 ClaimLens</h1>
        <h3 style="color: #6c757d; font-weight: 400; margin-top: 0;">Intelligent Customer Service</h3>
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
                response = requests.get(URL_STAGE1, params=params, timeout=10)
                response.raise_for_status()
                result = response.json()

                intent = result.get("Refund Intent", "Yes" if result.get("refund_intent") == 1 else "No")
                prob = float(result.get("Refund Intent Probability", result.get("probability", 0.0)))

                st.session_state.stage1_result = {
                    "intent": intent,
                    "probability": prob
                }

            except requests.exceptions.RequestException as e:
                st.error(f"Stage 1 API Request Failed: {e}")


if st.session_state.stage1_result:
    res1 = st.session_state.stage1_result
    st.markdown("---")
    st.subheader("📊 Stage 1 Intent Diagnosis")

    col1, col2 = st.columns(2)
    with col1:
        if res1["intent"] == "Yes":
            st.success(f"Predicted Claim Intent: **{res1['intent']}**")
        else:
            st.info(f"Predicted Claim Intent: **{res1['intent']}**")

    with col2:
        st.metric(label="Claim Intent Probability", value=f"{res1['probability']:.1%}")
        st.progress(min(max(res1["probability"], 0.0), 1.0))


if st.session_state.stage1_result and st.session_state.stage1_result["intent"] == "Yes":
    st.markdown("---")
    st.subheader("📋 Step 2: Additional Case Details")
    st.info("💡 Refund intent confirmed. Please provide further details (e.g. order number, item condition, specific issue) to proceed with the resolution.")

    with st.form("Stage 2 - Follow-up Details"):
        additional_info = st.text_area(
            "Please enter additional customer information",
            placeholder="e.g. Order #12345. The zipper is completely broken and torn off.",
            height=100
        )
        submitted_stage2 = st.form_submit_button("Predict the decision of refund", use_container_width=True)

        if submitted_stage2:
            if not additional_info.strip():
                st.warning("Please enter additional details.")
            else:
                combined_text = f"{st.session_state.first_message} {additional_info}"

                payload = {
                    "customer_message": combined_text
                }

                try:
                    with st.spinner("Processing second model inference..."):
                        response = requests.get(URL_STAGE2, params=payload, timeout=10)
                        response.raise_for_status()
                        result = response.json()

                        intent = result.get("Money Back", "Yes" if result.get("money_back") == 1 else "No")
                        prob = float(result.get("Chance of money loss", result.get("probability", 0.0)))

                        st.session_state.stage2_result = {
                            "money_back": intent,
                            "chance_of_loss": prob
                        }

                except requests.exceptions.RequestException as e:
                    st.error(f"Stage 2 API Request Failed: {e}")


if st.session_state.stage2_result:
    st.markdown("---")
    st.subheader("🏁 Stage 2 Final Resolution")
    res2 = st.session_state.stage2_result

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        if res2["money_back"] == "Yes":
            st.success(f"Money Back Approved: **{res2['money_back']}**")
        else:
            st.info(f"Money Back Approved: **{res2['money_back']}**")

    with col_res2:
        st.metric(label="Chance of Money Loss", value=f"{res2['chance_of_loss']:.1%}")
        st.progress(min(max(res2["chance_of_loss"], 0.0), 1.0))

    #with st.expander("🔍 View Raw API Response"):
    #    st.json(res2)
