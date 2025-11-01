#!/usr/bin/env python3
"""
FinSight – Real-Time Financial Dashboard
----------------------------------------
Stable, free-data dashboard with:
  • Exchange Rates (primary: Open ER-API, fallback: Frankfurter)
  • Central Bank / Market Rates (policy-rate feed via TE if API key set,
    else safe curated fallback; plus bond yields via Yahoo Finance)
  • Inflation (World Bank)
  • Market Indices, Commodities, Crypto (Yahoo Finance)
"""

import os
import time
import json
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="FinSight – Real-Time Financial Dashboard",
    page_icon="🏦",
    layout="wide"
)
st.title("🏦 FinSight – Real-Time Financial Dashboard")
st.caption("An intelligent dashboard visualizing live exchange rates, interest/market rates, inflation, and indices.")

# =============================
# HELPERS
# =============================
def _percent_change(curr, prev):
    try:
        if prev == 0:
            return 0.0
        return round((curr - prev) / prev * 100.0, 2)
    except Exception:
        return 0.0

def _safe_history(symbol, period="5d"):
    try:
        df = yf.Ticker(symbol).history(period=period)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# =============================
# SIDEBAR CONTROLS
# =============================
st.sidebar.header("⚙️ Controls")
base_currency = st.sidebar.selectbox("Base Currency (FX)", ["USD", "EUR", "GBP", "JPY"])
country_for_infl = st.sidebar.selectbox(
    "Inflation Country",
    ["Tunisia (TUN)", "United States (USA)", "Euro Area (EMU)", "France (FRA)"]
)
auto_refresh = st.sidebar.slider("Auto-refresh interval (seconds)", 30, 600, 120)

# =============================
# 1) EXCHANGE RATES (ROBUST)
# =============================
st.header("💱 Exchange Rates")

@st.cache_data(ttl=1800)
def get_exchange_rates(base="USD"):
    """
    Robust FX fetch:
      1) Primary: Open ER-API (https://open.er-api.com/v6/latest/{base})
      2) Fallback: Frankfurter (https://api.frankfurter.app/latest?from=BASE)
    Returns: (DataFrame[rates], date_str) or (empty_df, None)
    """
    # Primary
    try:
        url1 = f"https://open.er-api.com/v6/latest/{base}"
        r1 = requests.get(url1, timeout=10)
        if r1.status_code == 200:
            data = r1.json()
            if data.get("result") == "success" and "rates" in data:
                rates = data["rates"]
                date_str = data.get("time_last_update_utc", "")
                df = pd.DataFrame(list(rates.items()), columns=["Currency", "Rate"])
                return df.sort_values("Rate", ascending=False), date_str
    except Exception:
        pass

    # Fallback
    try:
        url2 = f"https://api.frankfurter.app/latest?from={base}"
        r2 = requests.get(url2, timeout=10)
        if r2.status_code == 200:
            data = r2.json()
            if "rates" in data:
                rates = data["rates"]
                date_str = data.get("date", "")
                df = pd.DataFrame(list(rates.items()), columns=["Currency", "Rate"])
                return df.sort_values("Rate", ascending=False), date_str
    except Exception:
        pass

    return pd.DataFrame(), None

fx_df, fx_date = get_exchange_rates(base_currency)
if fx_df.empty:
    st.warning("⚠️ Could not load exchange rates from both providers.")
else:
    st.markdown(f"**Last update:** {fx_date}")
    c1, c2 = st.columns(2)
    c1.bar_chart(fx_df.set_index("Currency")["Rate"].head(15))
    c2.bar_chart(fx_df.set_index("Currency")["Rate"].tail(15))

# =====================================================
# 2) CENTRAL BANK / MARKET RATES (STABLE + FALLBACKS)
# =====================================================
st.header("🏛️ Central Bank & Market Rates")

@st.cache_data(ttl=3600)
def get_policy_rates():
    """
    Policy rates:
      • If TRADING_ECON_API_KEY env var is set, query TradingEconomics JSON.
      • Else return a curated, safe fallback snapshot so UI never breaks.
    """
    te_key = os.getenv("TRADING_ECON_API_KEY", "").strip()
    if te_key:
        try:
            url = f"https://api.tradingeconomics.com/interest-rate?c={te_key}&format=json"
            r = requests.get(url, timeout=10)
            data = r.json()
            if isinstance(data, list) and data:
                keep = []
                for d in data:
                    # Normalize field names across TE variants
                    country = d.get("Country") or d.get("country") or "N/A"
                    last = d.get("Last") or d.get("last") or d.get("Value")
                    prev = d.get("Previous") or d.get("previous")
                    dt = d.get("DateTime") or d.get("datetime") or d.get("Date") or ""
                    keep.append({"Country/Area": country,
                                 "Policy Rate (%)": last,
                                 "Previous (%)": prev,
                                 "Date": dt})
                df = pd.DataFrame(keep)
                # Filter out obvious None rows
                df = df[df["Policy Rate (%)"].notnull()]
                if not df.empty:
                    return df.sort_values("Policy Rate (%)", ascending=False).head(15)
        except Exception:
            pass

    # Safe curated fallback (approximate, non-binding, update when needed)
    fallback = [
        {"Country/Area": "United States (Fed)",          "Policy Rate (%)": 5.50, "Previous (%)": 5.50, "Date": "—"},
        {"Country/Area": "Euro Area (ECB)",              "Policy Rate (%)": 4.50, "Previous (%)": 4.50, "Date": "—"},
        {"Country/Area": "United Kingdom (BoE)",         "Policy Rate (%)": 5.25, "Previous (%)": 5.25, "Date": "—"},
        {"Country/Area": "Tunisia (BCT)",                "Policy Rate (%)": 8.00, "Previous (%)": 8.00, "Date": "—"},
        {"Country/Area": "Canada (BoC)",                 "Policy Rate (%)": 5.00, "Previous (%)": 5.00, "Date": "—"},
        {"Country/Area": "Japan (BoJ)",                  "Policy Rate (%)": 0.10, "Previous (%)": 0.10, "Date": "—"},
        {"Country/Area": "Switzerland (SNB)",            "Policy Rate (%)": 1.50, "Previous (%)": 1.75, "Date": "—"},
        {"Country/Area": "Morocco (BAM)",                "Policy Rate (%)": 3.00, "Previous (%)": 3.00, "Date": "—"},
        {"Country/Area": "Egypt (CBE)",                  "Policy Rate (%)": 21.75, "Previous (%)": 20.75, "Date": "—"},
        {"Country/Area": "South Africa (SARB)",          "Policy Rate (%)": 8.25, "Previous (%)": 8.25, "Date": "—"},
    ]
    return pd.DataFrame(fallback)

policy_df = get_policy_rates()
if policy_df.empty:
    st.warning("⚠️ Could not load central bank policy rates.")
else:
    st.dataframe(policy_df, use_container_width=True)

# Bond yields (market reference) + commodities + crypto via Yahoo
@st.cache_data(ttl=1800)
def get_market_sheet():
    tickers = {
        "US 13W T-Bill (yield %)": "^IRX",   # ~multiplied by 100 already as index
        "US 10Y Treasury (yield %)": "^TNX", # same note: displayed as percent
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "FTSE 100": "^FTSE",
        "CAC 40": "^FCHI",
        "Nikkei 225": "^N225",
        "Gold (USD/oz)": "GC=F",
        "Oil WTI (USD/bbl)": "CL=F",
        "Silver (USD/oz)": "SI=F",
        "Bitcoin (USD)": "BTC-USD",
        "Ethereum (USD)": "ETH-USD",
    }
    rows = []
    for name, sym in tickers.items():
        df = _safe_history(sym, "5d")
        if not df.empty:
            curr = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2]) if len(df) > 1 else curr
            rows.append({
                "Instrument": name,
                "Symbol": sym,
                "Last": round(curr, 2),
                "Change (%)": _percent_change(curr, prev)
            })
    return pd.DataFrame(rows)

mkt_df = get_market_sheet()
if not mkt_df.empty:
    st.subheader("📊 Market Snapshot (Yields, Indices, Commodities, Crypto)")
    st.dataframe(mkt_df, use_container_width=True)

# =============================
# 3) INFLATION (WORLD BANK)
# =============================
st.header("📈 Inflation (World Bank)")

@st.cache_data(ttl=86400)
def get_inflation_wb(country_code="TUN"):
    """
    World Bank indicator FP.CPI.TOTL.ZG – Inflation, consumer prices (annual %).
    country_code e.g. 'TUN', 'USA', 'EMU', 'FRA'.
    """
    indicator = "FP.CPI.TOTL.ZG"
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json&per_page=60"
    try:
        r = requests.get(url, timeout=15)
        j = r.json()
        if isinstance(j, list) and len(j) > 1:
            rows = [{"Year": d["date"], "Inflation (%)": d["value"]}
                    for d in j[1] if d.get("value") is not None]
            df = pd.DataFrame(rows)
            return df.sort_values("Year")
    except Exception:
        pass
    return pd.DataFrame()

cc_map = {
    "Tunisia (TUN)": "TUN",
    "United States (USA)": "USA",
    "Euro Area (EMU)": "EMU",
    "France (FRA)": "FRA",
}
infl_df = get_inflation_wb(cc_map[country_for_infl])
if infl_df.empty:
    st.warning("⚠️ Could not load World Bank inflation series.")
else:
    st.line_chart(infl_df.set_index("Year")["Inflation (%)"])

# =============================
# KPI SUMMARY (optional eye candy)
# =============================
st.header("🧭 Quick KPIs")
k1, k2, k3 = st.columns(3)

# Best FX move (among first 30 for speed)
if not fx_df.empty:
    fx_sample = fx_df.head(30)
    top_fx = fx_sample.iloc[0]
    k1.metric("Largest FX rate (sample)", f"{top_fx['Currency']}", f"{top_fx['Rate']:.3f}")
else:
    k1.metric("Largest FX rate (sample)", "—", "—")

# Highest policy rate
if not policy_df.empty:
    top_policy = policy_df.sort_values("Policy Rate (%)", ascending=False).iloc[0]
    k2.metric("Highest Policy Rate", f"{top_policy['Country/Area']}", f"{top_policy['Policy Rate (%)']:.2f}%")
else:
    k2.metric("Highest Policy Rate", "—", "—")

# Best daily move in markets
if not mkt_df.empty:
    best_move = mkt_df.sort_values("Change (%)", ascending=False).iloc[0]
    k3.metric("Top Daily Gainer", best_move["Instrument"], f"{best_move['Change (%)']:.2f}%")
else:
    k3.metric("Top Daily Gainer", "—", "—")

# =============================
# AUTO REFRESH
# =============================
time.sleep(auto_refresh)
st.experimental_rerun()
