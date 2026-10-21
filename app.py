import streamlit as st
import requests
import os

st.set_page_config(
    page_title="ClaimLens - Two-Stage Refund Decision System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# One base URL for the API. Default is Cloud Run.
# To test against a local API: CLAIMLENS_API_URL=http://127.0.0.1:8000 uv run streamlit run app.py
API_URL = os.getenv("CLAIMLENS_API_URL", "https://claimlens-api-392994371879.europe-west3.run.app")
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
if "combined_message" not in st.session_state:
    st.session_state.combined_message = ""
if "error_stage" not in st.session_state:
    # "Model 1" | "Model 2"
    st.session_state.error_stage = None
if "chat" not in st.session_state:
    # customer conversation: list of {"role": "user" | "assistant", "text": str}
    st.session_state.chat = []


def reset_flow():
    st.session_state.first_message = ""
    st.session_state.stage1_result = None
    st.session_state.stage2_result = None
    st.session_state.step = "input"
    st.session_state.agent_reason = None
    st.session_state.combined_message = ""
    st.session_state.error_stage = None
    st.session_state.chat = []


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
# 3. Page header
# ----------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="margin-bottom: 0.25rem;">🔍 ClaimLens</h1>
        <h3 style="color: #6c757d; font-weight: 400; margin-top: 0;">Intelligen Customer Service</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# 4. Approved copy (customer side and company side)
# ----------------------------------------------------
CUSTOMER_ASK_CONTEXT = "Thanks for your message. To help you faster, please tell us more about what happened."
CUSTOMER_CHOICE = "You can add more details about your issue in the box below, or talk to an agent."
CUSTOMER_APPROVED = (
    "Your request has been pre-approved. Please wait 48 hours to receive more details by email. "
    "An agent might request additional information prior to final approval."
)
CUSTOMER_AGENT_TEXT = {
    "low_intent": f"Thanks for your message. {AGENT_LINE}",
    "customer_request": AGENT_LINE,
    "retry_no": AGENT_LINE,
    "model2_no": f"Thanks for the details. {AGENT_LINE}",
    "error": f"Sorry, something went wrong on our side. {AGENT_LINE}",
}

COMPANY_ASSIGNED = "An agent has been assigned to this customer chat."
COMPANY_AGENT_TEXT = {
    "low_intent": f"Low refund intent. {COMPANY_ASSIGNED}",
    "customer_request": COMPANY_ASSIGNED,
    "retry_no": f"Still no refund intent after more details. {COMPANY_ASSIGNED}",
    "model2_no": f"Human review recommended. {COMPANY_ASSIGNED}",
    "error": f"Technical error. No model result available. {COMPANY_ASSIGNED}",
}


# ----------------------------------------------------
# 5. API calls (one function per model)
# ----------------------------------------------------
API_ERRORS = (requests.exceptions.RequestException, KeyError, ValueError)


def call_model1(text):
    """Model 1: refund intent. Returns {"intent": "Yes"/"No", "probability": float}."""
    response = requests.get(URL_STAGE1, params={"customer_message": text}, timeout=30)
    response.raise_for_status()
    result = response.json()
    return {
        "intent": result["Refund Intent"],
        "probability": float(result["Refund Intent Probability"]),
    }


def call_model2(text):
    """Model 2: approved action. Returns {"intent": "Yes"/"No", "probability": float}."""
    response = requests.get(URL_STAGE2, params={"customer_message": text}, timeout=30)
    response.raise_for_status()
    result = response.json()
    return {
        "intent": result["Money Back"],
        "probability": float(result["Chance of money loss"]),
    }


# ----------------------------------------------------
# 6. Flow: every function sets the next step
# ----------------------------------------------------
def add_message(role, text):
    """Add one line to the customer chat. role is "user" or "assistant"."""
    st.session_state.chat.append({"role": role, "text": text})


def go_agent(reason):
    st.session_state.step = "agent"
    st.session_state.agent_reason = reason
    add_message("assistant", CUSTOMER_AGENT_TEXT[reason])


def go_error(stage):
    st.session_state.step = "error"
    st.session_state.error_stage = stage
    add_message("assistant", CUSTOMER_AGENT_TEXT["error"])


def run_model1(text):
    """Run Model 1, save the result, then pick the next step."""
    try:
        with st.spinner("Analysing your message..."):
            result = call_model1(text)
    except API_ERRORS:
        go_error("Model 1")
        return None
    st.session_state.stage1_result = result
    return result


def run_model2(text):
    """Run Model 2, save the result, then pick the next step."""
    try:
        with st.spinner("Analysing your details..."):
            result = call_model2(text)
    except API_ERRORS:
        go_error("Model 2")
        return None
    st.session_state.stage2_result = result
    if result["intent"] == "Yes":
        st.session_state.step = "approved"
        add_message("assistant", CUSTOMER_APPROVED)
    else:
        go_agent("model2_no")
    return result


def route_after_first_model1(result):
    """First run of Model 1: Yes asks for context. No depends on the probability."""
    if result["intent"] == "Yes":
        st.session_state.step = "context"
        add_message("assistant", CUSTOMER_ASK_CONTEXT)
    elif result["probability"] < LOW_INTENT_CUTOFF:
        go_agent("low_intent")
    else:
        st.session_state.step = "choice"
        add_message("assistant", CUSTOMER_CHOICE)


# ----------------------------------------------------
# 7. Company side helpers
# ----------------------------------------------------
def company_status(step):
    """Status line for the agent, picked from step and agent_reason."""
    if step == "input":
        return "Waiting for the customer message."
    if step == "context":
        return "Refund intent detected. Waiting for customer details."
    if step == "choice":
        return "Possible refund intent. Waiting for the customer to choose."
    if step == "approved":
        return (
            "Recommended for approval. The customer was told to expect an email within 48 hours. "
            "Review the case and email the customer within 48 hours."
        )
    if step == "agent":
        return COMPANY_AGENT_TEXT[st.session_state.agent_reason]
    return COMPANY_AGENT_TEXT["error"]


def show_model(title, result, placeholder):
    st.subheader(title)
    if result:
        st.metric("Probability", f"{result['probability']:.0%}")
        st.write(f"Prediction: **{result['intent']}**")
    else:
        st.caption(placeholder)


# ----------------------------------------------------
# 8. Layout: customer on the left, company on the right
# ----------------------------------------------------
step = st.session_state.step
left, right = st.columns(2, gap="large")

with left:
    st.header("Customer view")

    # The whole conversation stays on screen. Only the input below changes with the step.
    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.write(message["text"])

    if step == "input":
        with st.form("input_form"):
            opening_message = st.text_area(
                "Opening customer message",
                placeholder="e.g. My package arrived wet and the leather jacket is destroyed. Can I get a refund?",
                height=120,
            )
            submitted = st.form_submit_button("Analyse", use_container_width=True)

        if submitted:
            if not opening_message.strip():
                st.warning("Please enter a message first.")
            else:
                st.session_state.first_message = opening_message.strip()
                st.session_state.combined_message = st.session_state.first_message
                add_message("user", st.session_state.first_message)
                result = run_model1(st.session_state.first_message)
                if result:
                    route_after_first_model1(result)
                st.rerun()

    elif step == "context":
        with st.form("context_form"):
            details = st.text_area(
                "Further customer messages",
                placeholder="e.g. Order #12345. The zipper is completely broken and torn off.",
                height=100,
            )
            sent = st.form_submit_button("Send", use_container_width=True)

        if sent:
            if not details.strip():
                st.warning("Please enter some details.")
            else:
                combined = f"{st.session_state.first_message} {details.strip()}"
                st.session_state.combined_message = combined
                add_message("user", details.strip())
                run_model2(combined)
                st.rerun()

    elif step == "choice":
        with st.form("choice_form"):
            details = st.text_area(
                "Add more details",
                placeholder="e.g. The item arrived broken and I would like my money back.",
                height=100,
            )
            sent = st.form_submit_button("Send", use_container_width=True)
        talk_to_agent = st.button("Talk to an agent", use_container_width=True)

        if talk_to_agent:
            add_message("user", "Talk to an agent")
            go_agent("customer_request")
            st.rerun()

        if sent:
            if not details.strip():
                st.warning("Please enter some details, or talk to an agent.")
            else:
                # One retry only: this step is left after this click, whatever the answer is.
                combined = f"{st.session_state.first_message} {details.strip()}"
                st.session_state.combined_message = combined
                add_message("user", details.strip())
                result = run_model1(combined)
                if result and result["intent"] == "Yes":
                    run_model2(combined)
                elif result:
                    go_agent("retry_no")
                st.rerun()

with right:
    st.header("Company view")
    st.info(company_status(step))
    if step == "error":
        st.caption(f"Failed at: {st.session_state.error_stage}")

    show_model("Model 1: Refund intent", st.session_state.stage1_result, "Not run")
    show_model(
        "Model 2: Approved action",
        st.session_state.stage2_result,
        "Waiting for more context" if step == "context" else "Not run",
    )

    with st.expander("Messages used for these results"):
        st.write(st.session_state.combined_message or "No message yet.")
