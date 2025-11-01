import streamlit as st
import requests

st.set_page_config(page_title="FinSight Chatbot", page_icon="💬")

st.title("💬 FinSight Chatbot Assistant")
st.caption("Interact with FinSight's AI assistant for banking and finance insights.")

# Simple chat interface
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.chat_input("Type your message here...")

if user_input:
    # Display user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Here you'd call your backend FastAPI / local LLM endpoint
    try:
        # Example backend call
        response = requests.post(
            "http://localhost:8000/chat",
            json={"query": user_input},
            timeout=10
        )
        bot_reply = response.json().get("answer", "No response from server.")
    except Exception as e:
        bot_reply = f"⚠️ Connection error: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
