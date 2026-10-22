import html
import os
import time
from datetime import datetime

import requests
import streamlit as st

st.set_page_config(
    page_title="ClaimLens - Two-Stage Refund Decision System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# One base URL for the API. Default is Cloud Run.
# To test against a local API: CLAIMLENS_API_URL=http://127.0.0.1:8000 uv run streamlit run app.py
API_URL = os.getenv("CLAIMLENS_API_URL", "https://claimlens-api-392994371879.europe-west3.run.app")
URL_STAGE1 = f"{API_URL}/predict"
URL_STAGE2 = f"{API_URL}/more_context"

# Model 1 probability below this goes straight to an agent.
# The Yes/No label always comes from the API. This number only picks case 2a or 2b.
LOW_INTENT_CUTOFF = 0.25

# Model 2 says Yes and the probability is above this: approved without an agent click.
# Between the API threshold (78%) and this value, an agent clicks Approve or Request more information.
AUTO_APPROVE_CUTOFF = 0.95

# One sentence for every live handoff to an agent
AGENT_LINE = "An agent has been assigned to you and will be with you shortly."


if "first_message" not in st.session_state:
    st.session_state.first_message = ""
if "stage1_result" not in st.session_state:
    st.session_state.stage1_result = None
if "stage2_result" not in st.session_state:
    st.session_state.stage2_result = None
if "step" not in st.session_state:
    # input | context | choice | approved | preapproved | agent_approved | more_info | revisit | agent | error
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
if "request_ms" not in st.session_state:
    # total API time of the current analysis
    st.session_state.request_ms = 0
if "logged" not in st.session_state:
    # True once the finished analysis is saved in the test history
    st.session_state.logged = False
if "history" not in st.session_state:
    # finished analyses. reset_flow does NOT clear this list.
    st.session_state.history = []


def reset_flow():
    st.session_state.first_message = ""
    st.session_state.stage1_result = None
    st.session_state.stage2_result = None
    st.session_state.step = "input"
    st.session_state.agent_reason = None
    st.session_state.combined_message = ""
    st.session_state.error_stage = None
    st.session_state.chat = []
    st.session_state.request_ms = 0
    st.session_state.logged = False
    st.session_state.opening_text = ""


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

# ----------------------------------------------------
# 3. Style and header banner
# ----------------------------------------------------
# Colours of the page, buttons and tabs come from .streamlit/config.toml.
# This CSS only styles what the theme cannot: banner, status line and model cards.
STYLE = """
<style>
:root { --brick:#b9402c; --blue:#2f5d8c; --orange:#f26d4f; --green:#9ef0c4; --ok:#2a9a5b; --teal:#176b62; --ink:#222; --card:#fdfcfa; --line:#e4ddd3; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 1.4rem 3rem 1rem 3rem; max-width: none; }
[data-testid="stChatMessage"] { padding: .45rem .75rem; }
[data-testid="stVerticalBlock"] { gap: .75rem; }

.cl-banner { background:#10303f; border-radius:26px;
             padding:1.2rem 2.4rem 1.3rem 2.4rem; margin-bottom:1rem; }
.cl-banner-label { color:#a8e6cf; font-size:.85rem; letter-spacing:.28em; text-transform:uppercase; margin-bottom:.7rem; }
.cl-banner-title { color:var(--green); font-size:3.9rem; font-weight:800; line-height:1.1; margin:0 0 .3rem 0; }
.cl-banner-sub { color:#f1ede8; font-size:1.15rem; }

.cl-eyebrow { color:var(--teal); font-size:1rem; font-weight:700; letter-spacing:.18em; text-transform:uppercase; margin-bottom:.5rem; }

[data-testid="stAlert"] { background:#fff; border:1px solid var(--line); border-left:5px solid var(--teal); border-radius:12px; }
[data-testid="stAlertContainer"] { background:transparent; }
[data-testid="stAlert"] * { color:var(--ink) !important; }

.cl-model { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:.6rem .9rem .7rem .9rem; margin-bottom:.6rem; }
.cl-model-label { color:#9a948c; font-size:.85rem; margin-bottom:.45rem; }
.cl-model-box { background:#fff; border:1px solid var(--line); border-radius:10px; padding:.7rem 1rem; }
.cl-model-title { font-size:1rem; }
.cl-verdict { display:flex; align-items:baseline; gap:.9rem; flex-wrap:wrap; }
.cl-num { font-size:1.35rem; font-weight:800; line-height:1.2; margin-top:.35rem; }
.cl-conf { color:#9a948c; font-size:.85rem; }
.cl-pct { font-size:1.9rem; font-weight:800; line-height:1.15; }
.cl-red { color:var(--brick); }
.cl-green { color:var(--ok); }
.cl-blue { color:var(--blue); }
.cl-muted { color:#b9b2a8; }
.cl-bar { height:9px; background:#ece5db; border-radius:99px; margin:.6rem 0 .2rem 0; overflow:hidden; }
.cl-fill { height:100%; border-radius:99px; }
.cl-red-bg { background:var(--brick); }
.cl-green-bg { background:var(--ok); }
.cl-blue-bg { background:var(--blue); }
.cl-pred { font-size:1rem; margin-top:.5rem; }

.st-key-approve button { background:var(--ok); border-color:var(--ok); }
.st-key-approve button:hover { background:#237a4a; border-color:#237a4a; }
.st-key-revisit button { background:var(--brick); border-color:var(--brick); }
.st-key-revisit button:hover { background:#992f1e; border-color:#992f1e; }
.st-key-approve button *, .st-key-revisit button * { color:#fff !important; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

st.markdown(
    """
    <div class="cl-banner">
        <div class="cl-banner-title">🔎 ClaimLens</div>
        <div class="cl-banner-label">Customer message analysis</div>
        <div class="cl-banner-sub">Understand the request. Review the likely outcome.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# 4. Approved copy (customer side and company side)
# ----------------------------------------------------
CUSTOMER_ASK_CONTEXT = "Thanks for your message. To help you faster, please tell us more about what happened."
CUSTOMER_CHOICE = "You can add more details about your issue in the box below, or talk to an agent."
PRE_APPROVED = "Your request has been pre-approved."
CUSTOMER_PREAPPROVED = (
    f"{PRE_APPROVED} Please wait 48 hours to receive an email with more details. "
    "An agent might request additional information prior to final approval."
)
APPROVED = "Your request has been approved."
CUSTOMER_APPROVED = f"{APPROVED} You will receive an email with more details."
CUSTOMER_MORE_INFO = "An agent needs more information about your request. You will receive an email with the details."
CUSTOMER_REVISIT = "An agent is taking a second look at your request. You will receive an email with the details."
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


def timed(call, text):
    """Run one API call and add its duration to the request time of this analysis."""
    start = time.perf_counter()
    try:
        return call(text)
    finally:
        st.session_state.request_ms += int((time.perf_counter() - start) * 1000)


def run_model1(text):
    """Run Model 1, save the result, then pick the next step."""
    try:
        with st.spinner("Analysing your message..."):
            result = timed(call_model1, text)
    except API_ERRORS:
        go_error("Model 1")
        return None
    st.session_state.stage1_result = result
    return result


def run_model2(text):
    """Run Model 2, save the result, then pick the next step."""
    try:
        with st.spinner("Analysing your details..."):
            result = timed(call_model2, text)
    except API_ERRORS:
        go_error("Model 2")
        return None
    st.session_state.stage2_result = result
    if result["intent"] == "Yes" and result["probability"] > AUTO_APPROVE_CUTOFF:
        st.session_state.step = "approved"
        add_message("assistant", CUSTOMER_APPROVED)
    elif result["intent"] == "Yes":
        st.session_state.step = "preapproved"
        add_message("assistant", CUSTOMER_PREAPPROVED)
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
# 7. Helpers: company side, test history, example messages
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
            f"Approved. Model confidence is above {AUTO_APPROVE_CUTOFF:.0%}. "
            "The customer was told to expect an email with more details."
        )
    if step == "preapproved":
        return (
            "Recommended for approval. Review the case, then approve or request more information. "
            "The customer was told to expect an email within 48 hours."
        )
    if step == "agent_approved":
        return "Approved by an agent. The customer was told to expect an email with more details."
    if step == "more_info":
        return "More information requested. The customer was told to expect an email with the details."
    if step == "revisit":
        return "Decision revisited. An agent is reviewing this case. The customer was told to expect an email with the details."
    if step == "agent":
        return COMPANY_AGENT_TEXT[st.session_state.agent_reason]
    return COMPANY_AGENT_TEXT["error"]


def card_colour(result, default):
    """Green when the model says Yes. Otherwise the default colour of the card."""
    return "green" if result and result["intent"] == "Yes" else default


def model_card(label, title, result, colour, placeholder):
    """Card: "Title = YES" in big letters, the probability under it, "Model confidence" under the number.
    colour is "red" or "blue". Without a result the card shows a dash and the placeholder text."""
    if result:
        percent = round(result["probability"] * 100)
        intent = html.escape(str(result["intent"]).upper())
        box = (
            f'<div class="cl-verdict"><span class="cl-model-title">{title} =</span>'
            f'<span class="cl-pct cl-{colour}">{intent}</span></div>'
            f'<div class="cl-num cl-{colour}">{percent}%</div>'
            f'<div class="cl-conf">Model confidence</div>'
        )
        below_bar = ""
    else:
        percent = 0
        box = (
            f'<div class="cl-verdict"><span class="cl-model-title">{title}</span>'
            f'<span class="cl-pct cl-muted">&ndash;</span></div>'
        )
        below_bar = f'<div class="cl-pred">{placeholder}</div>'
    st.markdown(
        f'<div class="cl-model">'
        f'<div class="cl-model-label">{label}</div>'
        f'<div class="cl-model-box">{box}</div>'
        f'<div class="cl-bar"><div class="cl-fill cl-{colour}-bg" style="width:{percent}%"></div></div>'
        f'{below_bar}'
        f'</div>',
        unsafe_allow_html=True,
    )


HISTORY_OUTCOME = {
    "approved": "Approved automatically",
    "preapproved": "Pre-approved, waiting for an agent",
    "agent_approved": "Approved by an agent",
    "more_info": "More information requested",
    "revisit": "Decision revisited by an agent",
    "low_intent": "Agent: low refund intent",
    "customer_request": "Agent: customer request",
    "retry_no": "Agent: still no refund intent",
    "model2_no": "Agent: human review recommended",
}


def log_history_once(step):
    """Save a finished analysis in the test history. It runs once per analysis."""
    if step not in ("approved", "preapproved", "agent", "error") or st.session_state.logged:
        return

    if step == "error":
        outcome = f"Agent: technical error ({st.session_state.error_stage})"
    elif step in ("approved", "preapproved"):
        outcome = HISTORY_OUTCOME[step]
    else:
        outcome = HISTORY_OUTCOME[st.session_state.agent_reason]

    def show(result):
        return f"{result['intent']} ({result['probability']:.0%})" if result else "Not run"

    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Opening message": st.session_state.first_message,
        "Model 1": show(st.session_state.stage1_result),
        "Model 2": show(st.session_state.stage2_result),
        "Outcome": outcome,
        "Request time (ms)": st.session_state.request_ms,
    })
    st.session_state.logged = True


def highlight(text):
    """Show the approval sentences in green. Uses Streamlit colour syntax, no HTML."""
    for sentence in (PRE_APPROVED, APPROVED):
        text = text.replace(sentence, f":green-background[**{sentence}**]")
    return text


def agent_decides(new_step, customer_text):
    """The agent clicks a button on the company side. The customer gets the matching message."""
    st.session_state.step = new_step
    add_message("assistant", customer_text)
    if st.session_state.history:
        st.session_state.history[-1]["Outcome"] = HISTORY_OUTCOME[new_step]
    st.rerun()


FOOTER_TEXT = "ClaimLens team project | Le Wagon Data Science bootcamp"


# Demo prompts. Each one has a first message and a second message for the "Continue demo" button.
# The numbers are the scores seen in testing. The app always shows what the API returns.
# Delivery has no second message, so it has no Continue demo button.
#   Billing 1:      M1 71%, then M2 96%  -> approved automatically
#   Order issue:    M1 93%, then M2 57%  -> agent
#   Damaged item:   M1 44%, then M1 51% and M2 47% -> agent
#   Billing 2:      M1 44%, then M1 90% and M2 84% -> pre-approved
DEMO_PROMPTS = [
    ("💳 Billing 1",
     "Hey, I have an issue",
     "My bank statement says that I got charged twice for my order."),
    ("🛒 Order issue",
     "Hello! My item is wrong, I ordered a blue coat.",
     "I ordered a blue coat and received a green one."),
    ("📦 Damaged item",
     "Good evening, my item was damage upon arrival.",
     "I received my order last night and the item was completely destroyed when I opened the box. "
     "I want to exchange it."),
    ("💳 Billing 2",
     "I need to speak with customer service asap.",
     "You charged me 4 times the amount that i was supposed to pay"),
]
EXAMPLES = {label: first for label, first, _ in DEMO_PROMPTS}
EXAMPLES["🚚 Delivery"] = "My order has not arrived and the tracking has not moved for a week. I want my money back."
DEMO_SECOND = {first: second for _, first, second in DEMO_PROMPTS}


def use_example(text):
    st.session_state.opening_text = text


def fill_demo():
    """Continue demo: put the second message of the chosen prompt in the text box. Nothing runs until Send."""
    second = DEMO_SECOND.get(st.session_state.first_message)
    if second:
        box = "context_text" if st.session_state.step == "context" else "choice_text"
        st.session_state[box] = second


def send_context(details):
    """The customer sends more details after a Yes from Model 1. Model 2 reads all messages."""
    combined = f"{st.session_state.first_message} {details}"
    st.session_state.combined_message = combined
    add_message("user", details)
    run_model2(combined)
    st.rerun()


def send_choice(details):
    """The customer sends more details after a low No. One retry only: Model 1 runs again."""
    combined = f"{st.session_state.first_message} {details}"
    st.session_state.combined_message = combined
    add_message("user", details)
    result = run_model1(combined)
    if result and result["intent"] == "Yes":
        run_model2(combined)
    elif result:
        go_agent("retry_no")
    st.rerun()


# ----------------------------------------------------
# 8. Layout: two tabs. First tab has customer on the left and company on the right
# ----------------------------------------------------
step = st.session_state.step
log_history_once(step)

tab_analyse, tab_history = st.tabs(["✦ Analyse a message", "🕘 Test history"])

with tab_analyse:
    left, right = st.columns([5, 6], gap="large")

    with left:
        st.subheader("💬 Customer messages")
        if step == "input":
            st.caption("Try one of these to get started.")
            # Three buttons per row, so the labels stay readable on a laptop screen.
            items = list(EXAMPLES.items())
            for row_start in range(0, len(items), 3):
                example_columns = st.columns(3)
                for column, (label, text) in zip(example_columns, items[row_start:row_start + 3]):
                    column.button(label, on_click=use_example, args=(text,), use_container_width=True)

        with st.container(border=True):
            st.markdown('<div class="cl-eyebrow">Customer view</div>', unsafe_allow_html=True)

            # The whole conversation stays on screen. Only the input below changes with the step.
            for message in st.session_state.chat:
                with st.chat_message(message["role"]):
                    st.markdown(highlight(message["text"]))

            if step == "input":
                with st.form("input_form", border=False):
                    opening_message = st.text_area(
                        "Tell us what you need help with",
                        key="opening_text",
                        placeholder="e.g. My package arrived wet and the leather jacket is destroyed. Can I get a refund?",
                        height=120,
                    )
                    analyse_column, clear_column = st.columns([3, 2])
                    with analyse_column:
                        submitted = st.form_submit_button("✦ Analyse", type="primary", use_container_width=True)
                    with clear_column:
                        st.form_submit_button("↺ Clear", on_click=reset_flow, use_container_width=True)

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
                with st.form("context_form", border=False):
                    details = st.text_area(
                        "Further customer messages",
                        key="context_text",
                        help="Order number, condition of the item, or anything that explains the problem.",
                        placeholder="e.g. Order #12345. The zipper is completely broken and torn off.",
                        height=100,
                    )
                    sent = st.form_submit_button("Send", type="primary", use_container_width=True)

                if sent:
                    if not details.strip():
                        st.warning("Please enter some details.")
                    else:
                        send_context(details.strip())

            elif step == "choice":
                with st.form("choice_form", border=False):
                    details = st.text_area(
                        "Add more details",
                        key="choice_text",
                        help="Order number, condition of the item, or anything that explains the problem.",
                        placeholder="e.g. The item arrived broken and I would like my money back.",
                        height=100,
                    )
                    sent = st.form_submit_button("Send", type="primary", use_container_width=True)
                talk_to_agent = st.button("Talk to an agent", use_container_width=True)

                if talk_to_agent:
                    add_message("user", "Talk to an agent")
                    go_agent("customer_request")
                    st.rerun()

                if sent:
                    if not details.strip():
                        st.warning("Please enter some details, or talk to an agent.")
                    else:
                        send_choice(details.strip())

    with right:
        st.subheader("📊 Analysis results")
        st.markdown('<div class="cl-eyebrow">Company view</div>', unsafe_allow_html=True)
        st.info(company_status(step))
        if step == "error":
            st.caption(f"Failed at: {st.session_state.error_stage}")

        if step == "preapproved":
            approve_column, more_info_column = st.columns(2)
            with approve_column:
                approve_clicked = st.button("Approve", type="primary", key="approve", use_container_width=True)
            with more_info_column:
                more_info_clicked = st.button("Request more information", use_container_width=True)
            if approve_clicked:
                agent_decides("agent_approved", CUSTOMER_APPROVED)
            if more_info_clicked:
                agent_decides("more_info", CUSTOMER_MORE_INFO)

        if step == "approved":
            revisit_column, _ = st.columns(2)
            with revisit_column:
                revisit_clicked = st.button("Revisit decision", key="revisit", use_container_width=True)
            if revisit_clicked:
                agent_decides("revisit", CUSTOMER_REVISIT)

        with st.expander("Messages used for these results"):
            st.write(st.session_state.combined_message or "No message yet.")

        model_card("Model 1", "Predict Claim intent", st.session_state.stage1_result,
                   card_colour(st.session_state.stage1_result, "red"), "Not run")
        model_card(
            "Model 2",
            "Pre-approval recommendation",
            st.session_state.stage2_result,
            card_colour(st.session_state.stage2_result, "blue"),
            "Waiting for more context" if step == "context" else "Not run",
        )

        if st.session_state.request_ms:
            st.caption(f"Total request time: {st.session_state.request_ms} ms. Scores are rounded by the API.")
    # Demo helper: start again from the first screen. The test history is kept.
    restart_column, continue_column, _ = st.columns([1, 1, 3])
    with restart_column:
        st.button("↺ Restart demo", on_click=reset_flow, key="restart", use_container_width=True)
    with continue_column:
        if step in ("context", "choice") and DEMO_SECOND.get(st.session_state.first_message):
            st.button("▶ Continue demo", on_click=fill_demo, key="continue_demo", use_container_width=True)

with tab_history:
    st.subheader("🕘 Test history")
    if st.session_state.history:
        st.dataframe(list(reversed(st.session_state.history)), width="stretch", hide_index=True)
    else:
        st.caption("No analyses yet. Run a message in the first tab.")

st.markdown("---")
st.caption(FOOTER_TEXT)
