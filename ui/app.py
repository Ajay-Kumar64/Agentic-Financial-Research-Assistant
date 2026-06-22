# File: ui/app.py
import streamlit as st
import requests
import uuid

# Configure production workspace layout
st.set_page_config(
    page_title="Agentic Financial Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000/api/v1/chat"

# Initialize Session Thread Registry
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("📊 Agentic Financial Research Assistant")
st.caption("Production-grade LangGraph Engine with Real-Time Observability and Telemetry Tracking")

# Build Dashboard Workspace Grid Structure (Left: Conversation View | Right: Observability Panel)
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("Workspace Conversation")

    # Render persistent conversation window cards
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.write(chat["message"])
        else:
            with st.chat_message("assistant"):
                st.write(chat["message"])
                st.caption(f"⏱️ Tokens Consumed: {chat['tokens_used']}")

    # Handle incoming messaging inputs
    if user_query := st.chat_input(
            "Enter financial query (e.g., 'Compare RBI policy shifts or compute metric differences')..."):
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.chat_history.append({"role": "user", "message": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Executing LangGraph multi-step machine sequence..."):
                payload = {
                    "message": user_query,
                    "conversation_id": st.session_state.conversation_id
                }
                try:
                    response = requests.post(API_URL, json=payload, timeout=90)
                    if response.status_code == 200:
                        data = response.json()
                        final_ans = data.get("response", "")
                        tokens = data.get("tokens_used", 0)
                        traces = data.get("execution_trace", [])

                        st.write(final_ans)
                        st.caption(f"⏱️ Tokens Consumed: {tokens}")

                        # Store session outputs and latest traces for dashboard tracking panel
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "message": final_ans,
                            "tokens_used": tokens
                        })
                        st.session_state.latest_traces = traces
                    else:
                        st.error(f"Backend Server returned an error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend endpoint: {str(e)}")

        # Force interface re-render to update the tracing panel immediately
        st.rerun()

with right_col:
    st.subheader("🕵️ Execution Trace Viewer")
    st.markdown("Chronological telemetry metrics tracking agent loops and planning shifts.")

    if "latest_traces" in st.session_state and st.session_state.latest_traces:
        for trace in st.session_state.latest_traces:
            with st.expander(f"Step {trace['step_number']}: {trace['node_name'].upper()}", expanded=True):
                st.markdown(f"**Action Executed:** `{trace['action_taken']}`")
                if "timestamp" in trace.get("telemetry_metadata", {}):
                    st.caption(f"Logged at: {trace['telemetry_metadata']['timestamp']}")
    else:
        st.info("No active traces loaded. Submit a financial request to view the live agent reasoning path.")