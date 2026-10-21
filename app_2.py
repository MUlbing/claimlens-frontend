import streamlit as st
import requests

st.set_page_config(
    page_title="ClaimLens - Two-Stage Refund Decision System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 两个模型的 API 地址
URL_STAGE1 = "https://backboard-pulverize-perennial.ngrok-free.dev/predict"
URL_STAGE2 = "https://backboard-pulverize-perennial.ngrok-free.dev/predict"

# ----------------------------------------------------
# 1. 初始化 Session State (保留多轮交互数据)
# ----------------------------------------------------
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


# ----------------------------------------------------
# 2. 侧边栏配置
# ----------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search-property.png", width=64)
    st.title("ClaimLens Console")
    st.markdown("Refund intent recognition based on NLP pipeline processing")

    st.markdown("---")
    st.subheader("Model Pipeline")
    st.info("**Stage 1**: Intent Detection\n\n**Stage 2**: Case Resolution & Reasoning")

    threshold = st.slider("Intent Decision Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    if st.button("🔄 Reset Conversation", use_container_width=True):
        reset_flow()
        st.rerun()

    st.markdown("---")
    st.caption("© 2026 ClaimLens Team. All rights reserved.")

# ----------------------------------------------------
# 3. 阶段一：首轮消息输入与预测
# ----------------------------------------------------
st.title("🔍 ClaimLens - Customer Service Intelligence")

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
            st.session_state.stage2_result = None  # 重置第二阶段结果

            params = {
                "customer_message": customer_message,
                "threshold": threshold
            }

            try:
                response = requests.get(URL_STAGE1, params=params, timeout=10)
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

# 展示阶段一判定结果
if st.session_state.stage1_result:
    res1 = st.session_state.stage1_result
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if res1["intent"] == "Yes":
            st.error(f"Predicted Refund Intent: **{res1['intent']}**")
        else:
            st.success(f"Predicted Refund Intent: **{res1['intent']}**")
    with col2:
        st.metric(label="Refund Probability", value=f"{res1['probability']}")

# ----------------------------------------------------
# 4. 阶段二：当 Intent 为 'Yes' 时触发后续输入表单
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
                # 拼接两轮文本并构建请求体
                combined_text = f"{st.session_state.first_message} {additional_info}"
                payload = {
                    "initial_message": st.session_state.first_message,
                    "additional_message": additional_info,
                    "combined_message": combined_text
                }

                try:
                    with st.spinner("Processing second model inference..."):
                        #response_s2 = requests.post(URL_STAGE2, json=payload, timeout=15)
                        #response_s2.raise_for_status()
                        #st.session_state.stage2_result = response_s2.json()
                        params = payload['combined_message']
                        response = requests.get(URL_STAGE1, params=params, timeout=10)
                        response.raise_for_status()
                        result = response.json()

                        intent = result.get("Refund Intent", "Yes" if result.get("refund_intent") == 1 else "No")
                        prob = float(result.get("Refund Intent Probability", result.get("probability", 0.0)))

                        st.session_state.stage2_result = {
                                            "Money back": intent,
                                            "Probality of lossing money": prob
                                        }

                except requests.exceptions.RequestException as e:
                    st.error(f"Stage 2 API Request Failed: {e}")

# ----------------------------------------------------
# 5. 阶段二结果展示
# ----------------------------------------------------
if st.session_state.stage2_result:
    st.markdown("---")
    st.subheader("🏁 Stage 2 Final Resolution")
    res2 = st.session_state.stage2_result

    # 适配输出字段（若后端返回具体决策、分类或回复，可直接展示）
    st.json(res2)
