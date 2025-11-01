#!/usr/bin/env python3
"""
FinSight Multi-Page Streamlit App
---------------------------------
Navbar: Chat | Dashboard
"""

import streamlit as st

st.set_page_config(page_title="FinSight App", page_icon="🏦", layout="wide")

st.title("🏦 FinSight Banking Assistant")
st.caption("Select a section from the sidebar to explore FinSight’s capabilities.")

st.markdown("""
Welcome to **FinSight** — your intelligent banking AI platform.

Use the sidebar to navigate between:
- 💬 **Chatbot:** Ask banking, compliance, or financial questions.
- 📊 **Dashboard:** Explore real-time financial indicators.
""")
