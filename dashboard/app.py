#!/usr/bin/env python3
"""
FinSight – Real-Time Financial Intelligence Dashboard
====================================================

Displays:
- Live Exchange Rates (via exchangerate.host)
- Global Interest Rates (via TradingEconomics API)
- Inflation Data (via World Bank Open Data)
- Market Indices (via Yahoo Finance API)
"""

import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
import time

# ====================================================
# PAGE CONFIG
# ====================================================
st.set_page_config(page_title="FinSight Financial Dashboard", page_icon="💹", layout="wide")

st.title("🏦 FinSight – Real-Time Financial Dashboard")
st.caption("An intelligent dashboard visualizing live exchange rates, interest rates, inflation, and market indices.")

# ====================================================
# 1️⃣ EXCHANGE RATES SECTION
# ====================================================
st.header("💱 Exchange Rates")

@st.cache_data(ttl=1800)
def get_exchange_rates(base_currency="USD"):
    """Fetch exchange rates using exchangerate.host."""
    url = f"https://api.exchangerate.host/latest?base={base_currency}"
    resp = requests.get(url).json()
    df = pd.DataFrame(resp["rates"].items(), columns=["Currency", "Rate"])
    df["Date"] = resp["date"]
    return df, resp["date"]

base_currency = st.sidebar.selectbox("Base currency", ["USD", "EUR", "GBP", "JPY"])
rates_df, date = get_exchange_rates(base_currency)
st.write(f"**Last update:** {date}")

col1, col2 = st.columns(2)
col1.bar_chart(rates_df.set_index("Currency")["Rate"].head(15))
col2.bar_chart(rates_df.set_index("Currency")["Rate"].tail(15))

# ====================================================
# 2️⃣ INTEREST RATES SECTION
# ====================================================
st.header("🏦 Central Bank Interest Rates")

@st.cache_data(ttl=3600)
def get_interest_rates():
    """Fetch key interest rates (free from TradingEconomics public endpoint)."""
    url = "https://api.tradingeconomics.com/interest-rate"
    try:
        data = requests.get(url, timeout=10).json()
        df = pd.DataFrame(data)[["Country", "Last", "Previous", "DateTime"]]
        df.rename(columns={"Last": "Current Rate", "Previous": "Previous Rate"}, inplace=True)
        df.sort_values(by="Current Rate", ascending=False, inplace=True)
        return df.head(10)
    except Exception as e:
        st.warning(f"Could not load interest rates: {e}")
        return pd.DataFrame()

ir_df = get_interest_rates()
if not ir_df.empty:
    st.dataframe(ir_df, use_container_width=True)

# ====================================================
# 3️⃣ INFLATION DATA SECTION
# ====================================================
st.header("📈 Inflation Rates (World Bank)")

@st.cache_data(ttl=86400)
def get_inflation_data(country="Tunisia"):
    """Fetch inflation (CPI) data from World Bank API."""
    indicator = "FP.CPI.TOTL.ZG"  # Inflation, consumer prices (annual %)
    url = f"https://api.worldbank.org/v2/country/tn/indicator/{indicator}?format=json&per_page=10"
    data = requests.get(url).json()
    if len(data) > 1:
        records = [{"Year": d["date"], "Inflation (%)": d["value"]} for d in data[1] if d["value"] is not None]
        df = pd.DataFrame(records)
        return df
    return pd.DataFrame()

infl_df = get_inflation_data()
if not infl_df.empty:
    st.line_chart(infl_df.set_index("Year")["Inflation (%)"])

# ====================================================
# 4️⃣ MARKET INDICES (Yahoo Finance)
# ====================================================
st.header("📊 Global Market Indices")

@st.cache_data(ttl=1800)
def get_market_indices():
    """Fetch global market indices using Yahoo Finance."""
    indices = {
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ",
        "^DJI": "Dow Jones",
        "^FTSE": "FTSE 100",
        "^N225": "Nikkei 225",
        "^FCHI": "CAC 40",
        "^BVSP": "Bovespa"
    }

    data = []
    for symbol, name in indices.items():
        ticker = yf.Ticker(symbol)
        info = ticker.history(period="5d")
        if not info.empty:
            current = info["Close"].iloc[-1]
            previous = info["Close"].iloc[-2]
            change = ((current - previous) / previous) * 100
            data.append({"Index": name, "Current": round(current, 2), "Change (%)": round(change, 2)})

    return pd.DataFrame(data)

mkt_df = get_market_indices()
st.dataframe(mkt_df, use_container_width=True)

# ====================================================
# AUTO REFRESH
# ====================================================
refresh_time = st.sidebar.slider("Auto-refresh interval (seconds)", 30, 600, 120)
time.sleep(refresh_time)
st.experimental_rerun()
