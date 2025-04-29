import streamlit as st
from main import run  # Import the run function from your new backend
import pandas as pd

st.set_page_config(page_title="Sales Agent Assistant", layout="wide")
st.title("SDR AI Assistant")
st.caption("Your AI assistant for querying leads and drafting emails.")

# Session state for chat and UI
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you with your leads today?"}
    ]
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = []
if "current_dataframe" not in st.session_state:
    st.session_state.current_dataframe = None
if "current_email" not in st.session_state:
    st.session_state.current_email = {"subject": "", "body": "", "recipient": ""}

# Layout: 2 columns
col1, col2 = st.columns([3, 2])

# --- Left: Chat interface ---
with col1:
    st.header("Chat with your SDR Assistant")

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask your SDR assistant...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            agent_response = run(user_input, st.session_state.agent_memory)
            st.session_state.agent_memory = agent_response.get("memory", st.session_state.agent_memory)
        
        st.session_state.messages.append(
            {"role": "assistant", "content": agent_response["response"]}
        )
        with st.chat_message("assistant"):
            st.markdown(agent_response["response"])

        # Store tool output (e.g., DataFrame) for right column
        if agent_response.get("tool_output"):
            # If it's a DataFrame-like list of dicts, convert to DataFrame
            if isinstance(agent_response["tool_output"], list):
                st.session_state.current_dataframe = pd.DataFrame(agent_response["tool_output"])
            else:
                st.session_state.current_dataframe = agent_response["tool_output"]

        # Store email draft if present (customize as needed)
        if agent_response.get("actions"):
            for action in agent_response["actions"]:
                if action.get("tool") == "generate_sales_email":
                    st.session_state.current_email = {
                        "subject": agent_response.get("metadata", {}).get("email_subject", ""),
                        "body": agent_response.get("metadata", {}).get("email_body", ""),
                        "recipient": agent_response.get("metadata", {}).get("email_recipient", ""),
                    }

# --- Right: DataFrame and Email Draft ---
with col2:
    # DataFrame display
    with st.expander("Filtered Leads", expanded=True):
        st.subheader("Lead Search Results")
        df = st.session_state.current_dataframe
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df)
        else:
            st.info("No lead data to display yet. Ask me to find leads!")

    # Email draft display
    with st.expander("Email Draft", expanded=True):
        st.subheader("Current Email Draft")
        email = st.session_state.current_email
        if email["subject"]:
            st.markdown(f"**To:** {email.get('recipient', 'Recipient')}")
            st.markdown(f"**Subject:** {email['subject']}")
            st.text_area(
                "Email Body",
                value=email["body"],
                height=200,
                key="email_body_display",
                disabled=True,
            )
            recipient_email = st.text_input("Recipient Email Address")
            if st.button("Send Email") and recipient_email:
                # You can call your send_email function here if needed
                st.success(f"Dummy email sent to {recipient_email}!")
        else:
            st.info("No email draft yet. Ask me to create an email for a lead!")