from __future__ import annotations

import os
import json
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Simon Investment Brain",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_NAME = "SIMON INVESTMENT BRAIN"
APP_VERSION = "14.0"
DEFAULT_TICKER = "AAPL"

WATCHLIST = [
    "AAPL",
    "NVDA",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "AMD",
    "NFLX",
]


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #05070b;
    --panel: rgba(20,24,32,.72);
    --panel-strong: rgba(24,29,40,.90);
    --border: rgba(255,255,255,.095);
    --border-soft: rgba(255,255,255,.055);
    --text: #f5f7fb;
    --muted: #8992a3;
    --blue: #77a7ff;
    --blue2: #557cff;
    --green: #45d69a;
    --red: #ff6375;
    --yellow: #f4c95d;
    --purple: #9c8cff;
    --radius: 22px;
}

html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 78% -10%,
            rgba(75,120,255,.18),
            transparent 28%
        ),
        radial-gradient(
            circle at 10% 25%,
            rgba(88,70,255,.09),
            transparent 25%
        ),
        radial-gradient(
            circle at 85% 80%,
            rgba(0,190,255,.05),
            transparent 25%
        ),
        #05070b;
    color: var(--text);
}

/* ---------- hide chrome ---------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* ---------- sidebar ---------- */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            rgba(11,14,21,.97),
            rgba(5,7,11,.99)
        );
    border-right: 1px solid rgba(255,255,255,.06);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

/* ---------- general ---------- */

.block-container {
    max-width: 1600px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

/* ---------- glass ---------- */

.glass {
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.075),
            rgba(255,255,255,.025)
        );
    border: 1px solid var(--border);
    border-radius: var(--radius);
    backdrop-filter: blur(30px);
    -webkit-backdrop-filter: blur(30px);
    box-shadow:
        0 20px 70px rgba(0,0,0,.24),
        inset 0 1px 0 rgba(255,255,255,.035);
}

.glass-soft {
    background: rgba(255,255,255,.035);
    border: 1px solid var(--border-soft);
    border-radius: 18px;
}

/* ---------- brand ---------- */

.brand-row {
    display: flex;
    align-items: center;
    gap: 13px;
    margin-bottom: 7px;
}

.brand-icon {
    width: 42px;
    height: 42px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    background:
        linear-gradient(
            145deg,
            #78a8ff,
            #3b64dc
        );
    box-shadow:
        0 12px 35px rgba(73,119,255,.35);
    font-weight: 800;
    color: white;
}

.brand-title {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -1px;
}

.brand-subtitle {
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* ---------- top bar ---------- */

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 18px;
}

.topbar-left {
    color: #aeb6c6;
    font-size: 12px;
    letter-spacing: .4px;
}

.online {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(56,211,153,.09);
    border: 1px solid rgba(56,211,153,.18);
    color: #67e3ae;
    font-size: 11px;
    font-weight: 700;
}

.dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4ce09d;
    box-shadow: 0 0 13px #4ce09d;
}

/* ---------- hero ---------- */

.hero {
    position: relative;
    overflow: hidden;
    min-height: 300px;
    padding: 30px;
    border-radius: 30px;
    border: 1px solid rgba(255,255,255,.10);
    background:
        radial-gradient(
            circle at 88% 20%,
            rgba(70,116,255,.25),
            transparent 30%
        ),
        radial-gradient(
            circle at 55% 110%,
            rgba(86,64,255,.12),
            transparent 35%
        ),
        linear-gradient(
            145deg,
            rgba(28,31,41,.92),
            rgba(10,12,18,.88)
        );
    box-shadow:
        0 35px 100px rgba(0,0,0,.34),
        inset 0 1px 0 rgba(255,255,255,.06);
    backdrop-filter: blur(35px);
}

.hero::after {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -120px;
    top: -120px;
    border-radius: 50%;
    background: rgba(83,126,255,.12);
    filter: blur(55px);
}

.hero-kicker {
    color: #8e97aa;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-symbol {
    margin-top: 12px;
    font-size: 38px;
    font-weight: 850;
    letter-spacing: -2px;
}

.hero-name {
    color: #9ba3b4;
    margin-top: 3px;
    font-size: 13px;
}

.hero-price {
    margin-top: 27px;
    font-size: 48px;
    line-height: 1;
    font-weight: 850;
    letter-spacing: -2.7px;
}

.hero-change-up {
    color: var(--green);
}

.hero-change-down {
    color: var(--red);
}

.hero-change-flat {
    color: #a8b0be;
}

.hero-meta {
    margin-top: 18px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.pill {
    padding: 7px 11px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,.08);
    background: rgba(255,255,255,.045);
    color: #aab2c1;
    font-size: 11px;
}

/* ---------- conviction ---------- */

.conviction {
    height: 100%;
    padding: 22px;
    border-radius: 22px;
    background:
        radial-gradient(
            circle at 85% 15%,
            rgba(102,130,255,.18),
            transparent 35%
        ),
        rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.08);
}

.conviction-label {
    color: #8d96a7;
    font-size: 10px;
    letter-spacing: 1.8px;
    text-transform: uppercase;
}

.conviction-score {
    font-size: 50px;
    line-height: 1;
    font-weight: 850;
    margin-top: 14px;
}

.conviction-bar {
    height: 7px;
    background: rgba(255,255,255,.07);
    border-radius: 999px;
    margin-top: 18px;
    overflow: hidden;
}

.conviction-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(
        90deg,
        #527dff,
        #86a9ff
    );
}

.conviction-label-row {
    display: flex;
    justify-content: space-between;
    margin-top: 8px;
    font-size: 10px;
    color: #727b8c;
}

/* ---------- metric cards ---------- */

.metric-card {
    padding: 18px;
    min-height: 108px;
    border-radius: 19px;
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.055),
            rgba(255,255,255,.018)
        );
    border: 1px solid rgba(255,255,255,.075);
}

.metric-label {
    color: #747e90;
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 700;
}

.metric-value {
    margin-top: 12px;
    font-size: 23px;
    font-weight: 800;
    letter-spacing: -.7px;
}

.metric-caption {
    margin-top: 5px;
    color: #697282;
    font-size: 10px;
}

/* ---------- section ---------- */

.section-head {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin: 28px 0 12px;
}

.section-title {
    font-size: 17px;
    font-weight: 800;
    letter-spacing: -.4px;
}

.section-subtitle {
    color: #70798b;
    font-size: 10px;
    margin-top: 4px;
}

/* ---------- brain ---------- */

.brain {
    padding: 24px;
    border-radius: 24px;
    border: 1px solid rgba(119,150,255,.18);
    background:
        radial-gradient(
            circle at 92% 0%,
            rgba(93,125,255,.18),
            transparent 28%
        ),
        linear-gradient(
            145deg,
            rgba(47,56,87,.22),
            rgba(255,255,255,.025)
        );
}

.brain-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.brain-title {
    font-size: 20px;
    font-weight: 800;
}

.brain-status {
    color: #73dcae;
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.agent {
    padding: 17px;
    border-radius: 17px;
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.065);
    min-height: 150px;
}

.agent-name {
    color: #8791a3;
    font-size: 9px;
    letter-spacing: 1.4px;
    font-weight: 700;
    text-transform: uppercase;
}

.agent-score {
    margin-top: 9px;
    font-size: 27px;
    font-weight: 800;
}

.agent-line {
    margin-top: 10px;
    height: 5px;
    background: rgba(255,255,255,.065);
    border-radius: 999px;
}

.agent-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(
        90deg,
        #5c7dff,
        #94aaff
    );
}

.agent-note {
    color: #70798a;
    font-size: 10px;
    margin-top: 10px;
    line-height: 1.5;
}

/* ---------- thesis ---------- */

.thesis-card {
    padding: 23px;
    border-radius: 22px;
    background: rgba(255,255,255,.032);
    border: 1px solid rgba(255,255,255,.065);
}

.thesis-title {
    font-size: 14px;
    font-weight: 800;
    margin-bottom: 12px;
}

.thesis-text {
    color: #a0a8b7;
    line-height: 1.7;
    font-size: 12px;
}

/* ---------- valuation ---------- */

.scenario {
    padding: 18px;
    border-radius: 17px;
    background: rgba(255,255,255,.032);
    border: 1px solid rgba(255,255,255,.065);
}

.scenario-name {
    color: #7c8698;
    font-size: 9px;
    letter-spacing: 1.3px;
    text-transform: uppercase;
}

.scenario-price {
    margin-top: 9px;
    font-size: 25px;
    font-weight: 800;
}

.scenario-up {
    color: var(--green);
}

.scenario-down {
    color: var(--red);
}

.scenario-neutral {
    color: #c2c9d5;
}

/* ---------- risk ---------- */

.risk-card {
    padding: 20px;
    border-radius: 20px;
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.07);
}

.risk-value {
    font-size: 34px;
    font-weight: 850;
}

.risk-low {
    color: var(--green);
}

.risk-mid {
    color: var(--yellow);
}

.risk-high {
    color: var(--red);
}

/* ---------- sidebar cards ---------- */

.sidebar-title {
    color: #6e7788;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.7px;
    text-transform: uppercase;
    margin-top: 20px;
    margin-bottom: 10px;
}

.watch-button {
    margin-bottom: 7px;
}

/* ---------- streamlit widgets ---------- */

div[data-testid="stButton"] > button {
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,.08);
    background: rgba(255,255,255,.045);
    color: #d9dee8;
    transition: all .18s ease;
}

div[data-testid="stButton"] > button:hover {
    border-color: rgba(115,151,255,.45);
    background: rgba(90,120,255,.11);
    transform: translateY(-1px);
}

div[data-testid="stButton"] > button[kind="primary"] {
    background:
        linear-gradient(
            135deg,
            #5e82ff,
            #4268e0
        );
    border: none;
    color: white;
    box-shadow: 0 12px 30px rgba(66,104,224,.25);
}

div[data-baseweb="select"] > div {
    background: rgba(255,255,255,.045);
    border-color: rgba(255,255,255,.08);
    border-radius: 13px;
}

div[data-baseweb="input"] > div {
    background: rgba(255,255,255,.045);
    border-color: rgba(255,255,255,.08);
    border-radius: 13px;
}

.stTextInput input {
    color: white;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    padding: 0 16px;
    border-radius: 12px;
    color: #727b8c;
}

.stTabs [aria-selected="true"] {
    background: rgba(255,255,255,.065);
    color: #f3f5fa !important;
}

div[data-testid="stMetric"] {
    background: transparent;
}

div[data-testid="stMetricValue"] {
    font-weight: 800;
}

.stAlert {
    border-radius: 16px;
}

hr {
    border-color: rgba(255,255,255,.06);
}

/* ---------- data source badge ---------- */

.data-badge-live {
    color: #5de0a4;
    background: rgba(60,220,150,.08);
    border: 1px solid rgba(60,220,150,.15);
}

.data-badge-demo {
    color: #f3c85c;
    background: rgba(243,200,92,.08);
    border: 1px solid rgba(243,200,92,.15);
}

.data-badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ---------- footer ---------- */

.footer {
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,.06);
    color: #596273;
    font-size: 9px;
    line-height: 1.7;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# UTILS
# ============================================================

def safe_float(value, default=None):
    try:
        if value is None:
            return default
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def fmt_money(value, digits=2):
    value = safe_float(value)
    if value is None:
        return "—"
    return f"${value:,.{digits}f}"


def fmt_percent(value, digits=2):
    value = safe_float(value)
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def fmt_large(value):
    value = safe_float(value)
    if value is None:
        return "—"

    value = abs(value)

    if value >= 1e12:
        return f"${value / 1e12:.2f}T"

    if value >= 1e9:
        return f"${value / 1e9:.2f}B"

    if value >= 1e6:
        return f"${value / 1e6:.2f}M"

    return f"${value:,.0f}"


def clamp(value, low=0, high=100):
    return max(low, min(high, float(value)))


def score_class(score):
    score = safe_float(score, 50)
    if score >= 70:
        return "risk-low"
    if score >= 45:
        return "risk-mid"
    return "risk-high"


def get_secret(name, default=None):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass

    return os.getenv(name, default)


# ============================================================
# SESSION STATE
# ============================================================

if "ticker" not in st.session_state:
    st.session_state.ticker = DEFAULT_TICKER

if "research" not in st.session_state:
    st.session_state.research = None

if "last_loaded" not in st.session_state:
    st.session_state.last_loaded = None


# ============================================================
# DEMO DATA FALLBACK
# ============================================================

def demo_profile(ticker: str) -> Dict[str, Any]:

    profiles = {
        "AAPL": {
            "name": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "market_cap": 3.45e12,
            "pe": 31.2,
            "forward_pe": 28.5,
            "price_to_book": 49.2,
            "revenue_growth": 0.074,
            "earnings_growth": 0.116,
            "gross_margin": 0.466,
            "operating_margin": 0.315,
            "roe": 1.56,
            "debt_to_equity": 1.52,
            "free_cash_flow": 1.08e11,
            "cash": 6.7e10,
            "debt": 1.2e11,
        },
        "NVDA": {
            "name": "NVIDIA Corporation",
            "sector": "Technology",
            "industry": "Semiconductors",
            "market_cap": 4.0e12,
            "pe": 47.0,
            "forward_pe": 32.0,
            "price_to_book": 35.0,
            "revenue_growth": 0.55,
            "earnings_growth": 0.65,
            "gross_margin": 0.73,
            "operating_margin": 0.62,
            "roe": 1.05,
            "debt_to_equity": 0.18,
            "free_cash_flow": 6.0e10,
            "cash": 4.2e10,
            "debt": 1.1e10,
        },
        "MSFT": {
            "name": "Microsoft Corporation",
            "sector": "Technology",
            "industry": "Software",
            "market_cap": 3.8e12,
            "pe": 36.0,
            "forward_pe": 30.0,
            "price_to_book": 12.0,
            "revenue_growth": 0.14,
            "earnings_growth": 0.16,
            "gross_margin": 0.69,
            "operating_margin": 0.45,
            "roe": 0.34,
            "debt_to_equity": 0.32,
            "free_cash_flow": 7.0e10,
            "cash": 7.8e10,
            "debt": 7.2e10,
        },
    }

    base = profiles.get(
        ticker,
        {
            "name": ticker,
            "sector": "Technology",
            "industry": "Technology",
            "market_cap": 8e11,
            "pe": 27,
            "forward_pe": 24,
            "price_to_book": 7,
            "revenue_growth": .12,
            "earnings_growth": .14,
            "gross_margin": .55,
            "operating_margin": .25,
            "roe": .30,
            "debt_to_equity": .45,
            "free_cash_flow": 2e10,
            "cash": 3e10,
            "debt": 2e10,
        },
    )

    return base


def demo_history(ticker: str, days: int = 420) -> pd.DataFrame:

    seed = sum(ord(c) for c in ticker)
    rng = np.random.default_rng(seed)

    profile = demo_profile(ticker)

    if ticker == "NVDA":
        start = 140
        drift = 0.0012
    elif ticker == "MSFT":
        start = 420
        drift = 0.00045
    elif ticker == "AAPL":
        start = 185
        drift = 0.00035
    else:
        start = 100
        drift = 0.00025

    dates = pd.bdate_range(
        end=pd.Timestamp.today(),
        periods=days,
    )

    returns = rng.normal(
        loc=drift,
        scale=.018,
        size=len(dates),
    )

    prices = start * np.exp(
        np.cumsum(returns)
    )

    volume = rng.integers(
        10_000_000,
        80_000_000,
        len(dates),
    )

    df = pd.DataFrame(
        {
            "Open": prices * (
                1 + rng.normal(0, .006, len(prices))
            ),
            "High": prices * (
                1 + abs(rng.normal(0, .012, len(prices)))
            ),
            "Low": prices * (
                1 - abs(rng.normal(0, .012, len(prices)))
            ),
            "Close": prices,
            "Adj Close": prices,
            "Volume": volume,
        },
        index=dates,
    )

    return df


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=180, show_spinner=False)
def fetch_history(
    ticker: str,
    period: str = "1y",
) -> Tuple[pd.DataFrame, str, Optional[str]]:

    if yf is None:
        return (
            demo_history(ticker),
            "DEMO",
            "yfinance is not installed",
        )

    try:
        t = yf.Ticker(ticker)

        df = t.history(
            period=period,
            interval="1d",
            auto_adjust=False,
        )

        if df is None or df.empty:
            raise RuntimeError(
                "Yahoo Finance returned no history."
            )

        df = df.copy()

        df.index = pd.to_datetime(
            df.index
        )

        df = df[
            [
                c for c in [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Volume",
                ]
                if c in df.columns
            ]
        ]

        df = df.dropna(
            subset=["Close"]
        )

        if len(df) < 30:
            raise RuntimeError(
                "Not enough market history."
            )

        return df, "LIVE", None

    except Exception as exc:

        return (
            demo_history(ticker),
            "DEMO",
            str(exc),
        )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_profile(
    ticker: str,
) -> Tuple[Dict[str, Any], str, Optional[str]]:

    fallback = demo_profile(ticker)

    if yf is None:
        return fallback, "DEMO", "yfinance unavailable"

    try:

        t = yf.Ticker(ticker)

        info = {}

        try:
            info = t.info or {}
        except Exception:
            info = {}

        if not info:
            return fallback, "DEMO", "No fundamental profile returned"

        result = {
            "name": info.get(
                "longName"
            ) or info.get(
                "shortName"
            ) or ticker,

            "sector": info.get(
                "sector"
            ),

            "industry": info.get(
                "industry"
            ),

            "market_cap": info.get(
                "marketCap"
            ),

            "pe": info.get(
                "trailingPE"
            ),

            "forward_pe": info.get(
                "forwardPE"
            ),

            "price_to_book": info.get(
                "priceToBook"
            ),

            "revenue_growth": info.get(
                "revenueGrowth"
            ),

            "earnings_growth": info.get(
                "earningsGrowth"
            ),

            "gross_margin": info.get(
                "grossMargins"
            ),

            "operating_margin": info.get(
                "operatingMargins"
            ),

            "roe": info.get(
                "returnOnEquity"
            ),

            "debt_to_equity": info.get(
                "debtToEquity"
            ),

            "free_cash_flow": info.get(
                "freeCashflow"
            ),

            "cash": info.get(
                "totalCash"
            ),

            "debt": info.get(
                "totalDebt"
            ),

            "website": info.get(
                "website"
            ),

            "country": info.get(
                "country"
            ),

            "beta": info.get(
                "beta"
            ),

            "dividend_yield": info.get(
                "dividendYield"
            ),
        }

        # If essentially everything failed, fallback.
        useful = [
            result.get("market_cap"),
            result.get("pe"),
            result.get("revenue_growth"),
            result.get("roe"),
        ]

        if not any(v is not None for v in useful):
            return fallback, "DEMO", "Incomplete fundamental response"

        for key, value in fallback.items():

            if result.get(key) is None:
                result[key] = value

        return result, "LIVE", None

    except Exception as exc:

        return fallback, "DEMO", str(exc)


@st.cache_data(ttl=120, show_spinner=False)
def fetch_quote(
    ticker: str,
    history: pd.DataFrame,
) -> Dict[str, Any]:

    last_price = safe_float(
        history["Close"].iloc[-1]
    )

    previous = safe_float(
        history["Close"].iloc[-2]
    )

    change = (
        last_price - previous
        if last_price is not None
        and previous is not None
        else None
    )

    change_pct = (
        change / previous
        if change is not None
        and previous
        else None
    )

    volume = None

    if "Volume" in history.columns:
        volume = safe_float(
            history["Volume"].iloc[-1]
        )

    return {
        "price": last_price,
        "previous_close": previous,
        "change": change,
        "change_percent": change_pct,
        "volume": volume,
    }


# ============================================================
# TECHNICAL ENGINE
# ============================================================

def calculate_technicals(
    df: pd.DataFrame,
) -> pd.DataFrame:

    data = df.copy()

    close = data["Close"]

    data["SMA20"] = close.rolling(
        20
    ).mean()

    data["SMA50"] = close.rolling(
        50
    ).mean()

    data["SMA200"] = close.rolling(
        200
    ).mean()

    data["EMA12"] = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    data["EMA26"] = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    data["MACD"] = (
        data["EMA12"]
        - data["EMA26"]
    )

    data["MACD_SIGNAL"] = (
        data["MACD"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        14
    ).mean()

    avg_loss = loss.rolling(
        14
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan,
        )
    )

    data["RSI"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    data["VOL20"] = (
        data["Volume"]
        .rolling(20)
        .mean()
        if "Volume" in data.columns
        else np.nan
    )

    data["ATR14"] = (
        data["High"]
        - data["Low"]
    ).rolling(14).mean()

    return data


def technical_snapshot(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    if df.empty:
        return {}

    latest = df.iloc[-1]

    price = safe_float(
        latest.get("Close")
    )

    sma20 = safe_float(
        latest.get("SMA20")
    )

    sma50 = safe_float(
        latest.get("SMA50")
    )

    sma200 = safe_float(
        latest.get("SMA200")
    )

    rsi = safe_float(
        latest.get("RSI")
    )

    macd = safe_float(
        latest.get("MACD")
    )

    macd_signal = safe_float(
        latest.get("MACD_SIGNAL")
    )

    trend = "NEUTRAL"

    if (
        price is not None
        and sma50 is not None
        and sma200 is not None
    ):

        if price > sma50 > sma200:
            trend = "BULLISH"

        elif price < sma50 < sma200:
            trend = "BEARISH"

    momentum = "NEUTRAL"

    if rsi is not None:

        if rsi >= 70:
            momentum = "OVERBOUGHT"

        elif rsi <= 30:
            momentum = "OVERSOLD"

        elif rsi >= 55:
            momentum = "POSITIVE"

        elif rsi <= 45:
            momentum = "NEGATIVE"

    macd_state = "NEUTRAL"

    if (
        macd is not None
        and macd_signal is not None
    ):

        if macd > macd_signal:
            macd_state = "POSITIVE"
        else:
            macd_state = "NEGATIVE"

    returns = df["Close"].pct_change()

    volatility = safe_float(
        returns.tail(30).std()
    )

    return {
        "price": price,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "trend": trend,
        "momentum": momentum,
        "macd_state": macd_state,
        "volatility_30d": volatility,
    }


# ============================================================
# FUNDAMENTAL ENGINE
# ============================================================

def fundamental_scores(
    fundamentals: Dict[str, Any],
) -> Dict[str, float]:

    revenue_growth = safe_float(
        fundamentals.get("revenue_growth"),
        .08,
    )

    earnings_growth = safe_float(
        fundamentals.get("earnings_growth"),
        .08,
    )

    gross_margin = safe_float(
        fundamentals.get("gross_margin"),
        .35,
    )

    operating_margin = safe_float(
        fundamentals.get("operating_margin"),
        .15,
    )

    roe = safe_float(
        fundamentals.get("roe"),
        .15,
    )

    debt = safe_float(
        fundamentals.get("debt_to_equity"),
        70,
    )

    pe = safe_float(
        fundamentals.get("pe"),
        25,
    )

    # Growth
    growth = clamp(
        50
        + revenue_growth * 180
        + earnings_growth * 100
    )

    # Quality
    quality = clamp(
        35
        + gross_margin * 40
        + operating_margin * 55
        + min(roe, 1.0) * 25
    )

    # Balance sheet
    balance = clamp(
        90
        - min(debt, 300) * .18
    )

    # Valuation
    valuation = clamp(
        90
        - max(pe - 15, 0) * 2.3
    )

    return {
        "growth": growth,
        "quality": quality,
        "balance": balance,
        "valuation": valuation,
    }


# ============================================================
# INVESTMENT BRAIN
# ============================================================

def compute_brain(
    ticker: str,
    fundamentals: Dict[str, Any],
    technicals: Dict[str, Any],
) -> Dict[str, Any]:

    fs = fundamental_scores(
        fundamentals
    )

    momentum_score = 50

    trend = technicals.get(
        "trend"
    )

    momentum = technicals.get(
        "momentum"
    )

    rsi = safe_float(
        technicals.get("rsi")
    )

    if trend == "BULLISH":
        momentum_score += 22
    elif trend == "BEARISH":
        momentum_score -= 22

    if momentum == "POSITIVE":
        momentum_score += 12
    elif momentum == "NEGATIVE":
        momentum_score -= 12
    elif momentum == "OVERBOUGHT":
        momentum_score += 5
    elif momentum == "OVERSOLD":
        momentum_score -= 2

    if rsi is not None:

        if rsi > 75:
            momentum_score -= 8

        elif rsi < 25:
            momentum_score += 6

    momentum_score = clamp(
        momentum_score
    )

    # AI-style investment conviction
    conviction = (
        fs["valuation"] * .20
        + fs["quality"] * .24
        + fs["growth"] * .22
        + momentum_score * .18
        + fs["balance"] * .16
    )

    conviction = clamp(
        conviction
    )

    # Risk
    risk = (
        100
        - (
            fs["quality"] * .30
            + fs["balance"] * .25
            + fs["valuation"] * .20
            + momentum_score * .25
        )
    )

    risk = clamp(
        risk
    )

    # Decision
    if conviction >= 78:
        decision = "ACCUMULATE"
    elif conviction >= 66:
        decision = "WATCH / BUILD"
    elif conviction >= 52:
        decision = "HOLD / WAIT"
    else:
        decision = "AVOID / REDUCE"

    return {
        "valuation": round(
            fs["valuation"]
        ),
        "quality": round(
            fs["quality"]
        ),
        "growth": round(
            fs["growth"]
        ),
        "momentum": round(
            momentum_score
        ),
        "balance": round(
            fs["balance"]
        ),
        "conviction": round(
            conviction
        ),
        "risk": round(
            risk
        ),
        "decision": decision,
    }


# ============================================================
# VALUATION ENGINE
# ============================================================

def valuation_scenarios(
    price: Optional[float],
    fundamentals: Dict[str, Any],
) -> Dict[str, float]:

    price = safe_float(
        price,
        100,
    )

    growth = safe_float(
        fundamentals.get(
            "revenue_growth"
        ),
        .10,
    )

    earnings_growth = safe_float(
        fundamentals.get(
            "earnings_growth"
        ),
        .10,
    )

    pe = safe_float(
        fundamentals.get(
            "pe"
        ),
        25,
    )

    # Simplified scenario framework.
    # This is not a production DCF.
    growth_factor = (
        1
        + min(
            max(
                growth,
                -.20,
            ),
            .60,
        )
    )

    earnings_factor = (
        1
        + min(
            max(
                earnings_growth,
                -.30,
            ),
            .70,
        )
    )

    quality_factor = (
        min(
            max(
                pe / 25,
                .65,
            ),
            1.6,
        )
    )

    base = price * (
        .55 * growth_factor
        + .45 * earnings_factor
    ) * (
        1 / max(
            quality_factor ** .12,
            .85,
        )
    )

    bull = base * 1.18
    bear = base * .72

    return {
        "bear": bear,
        "base": base,
        "bull": bull,
    }


# ============================================================
# AI LAYER
# ============================================================

def ai_config() -> Dict[str, Any]:

    key = get_secret(
        "OPENAI_API_KEY"
    )

    model = get_secret(
        "OPENAI_MODEL",
        "gpt-4o-mini",
    )

    base_url = get_secret(
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1",
    )

    return {
        "configured": bool(key),
        "api_key": key,
        "model": model,
        "base_url": base_url.rstrip("/"),
    }


def local_ai_memo(
    ticker: str,
    fundamentals: Dict[str, Any],
    technicals: Dict[str, Any],
    brain: Dict[str, Any],
    scenarios: Dict[str, float],
) -> Dict[str, str]:

    growth = brain["growth"]
    quality = brain["quality"]
    valuation = brain["valuation"]
    momentum = brain["momentum"]
    risk = brain["risk"]

    decision = brain["decision"]

    trend = technicals.get(
        "trend",
        "NEUTRAL",
    )

    revenue_growth = fmt_percent(
        fundamentals.get(
            "revenue_growth"
        )
    )

    pe = fundamentals.get(
        "pe"
    )

    if growth >= 75:
        growth_comment = (
            f"{ticker} 当前增长质量较强，"
            f"收入增长约 {revenue_growth}，"
            "市场仍可能给予成长溢价。"
        )
    elif growth >= 55:
        growth_comment = (
            f"{ticker} 增长处于中等偏上水平，"
            "核心变量是未来盈利增速能否继续兑现。"
        )
    else:
        growth_comment = (
            f"{ticker} 当前增长并不突出，"
            "后续估值扩张需要更强的基本面催化。"
        )

    if quality >= 78:
        quality_comment = (
            "商业质量较强，利润率、资本效率或现金流能力"
            "构成主要护城河。"
        )
    elif quality >= 60:
        quality_comment = (
            "商业质量处于中上水平，但仍需要观察"
            "利润率与资本回报能否持续。"
        )
    else:
        quality_comment = (
            "商业质量存在明显不确定性，"
            "不宜仅依靠故事驱动估值。"
        )

    if valuation >= 75:
        valuation_comment = (
            "当前估值在模型框架下具有相对吸引力，"
            "安全边际尚可。"
        )
    elif valuation >= 55:
        valuation_comment = (
            "当前估值大致合理，"
            "未来收益更多依赖盈利增长而非估值扩张。"
        )
    else:
        valuation_comment = (
            f"当前估值偏贵"
            + (
                f"，Trailing P/E 约 {pe:.1f}x。"
                if pe is not None
                else "。"
            )
        )

    if trend == "BULLISH":
        technical_comment = (
            "价格结构处于多头状态，趋势对中期投资者较友好。"
        )
    elif trend == "BEARISH":
        technical_comment = (
            "价格结构偏弱，基本面再好也需要防范"
            "估值继续压缩带来的回撤。"
        )
    else:
        technical_comment = (
            "价格结构处于中性区域，等待趋势进一步确认。"
        )

    thesis = (
        f"{ticker} 的核心投资逻辑来自"
        f"「增长 × 商业质量 × 估值」的组合。"
        f"当前 AI Conviction 为 {brain['conviction']}/100，"
        f"模型给出的初步行动评级为 {decision}。"
    )

    risk_text = (
        f"主要风险等级为 {risk}/100。"
        "需要重点关注估值压缩、盈利预期下修、"
        "宏观流动性变化以及公司特定事件。"
    )

    return {
        "thesis": thesis,
        "growth": growth_comment,
        "quality": quality_comment,
        "valuation": valuation_comment,
        "technical": technical_comment,
        "risk": risk_text,
        "decision": decision,
        "committee": (
            f"Investment Committee：{decision}。\n\n"
            f"核心判断：{thesis}\n\n"
            f"Bull Case：{fmt_money(scenarios['bull'])}\n"
            f"Base Case：{fmt_money(scenarios['base'])}\n"
            f"Bear Case：{fmt_money(scenarios['bear'])}\n\n"
            "该结果是研究辅助框架，不构成投资建议。"
        ),
    }


def call_llm(
    ticker: str,
    context: Dict[str, Any],
) -> Optional[str]:

    config = ai_config()

    if not config["configured"]:
        return None

    system_prompt = """
You are Simon Investment Brain, an institutional-grade US equity
research committee.

Your job is NOT to blindly predict stock prices.

Analyze the company using five lenses:

1. Value
2. Business Quality
3. Growth / First Principles
4. Market / Momentum
5. Risk

Then produce:

- Investment thesis
- Bull case
- Base case
- Bear case
- Key catalysts
- Key risks
- What would invalidate the thesis
- Conviction score 0-100
- Decision: ACCUMULATE / WATCH / HOLD / REDUCE

Do not claim access to data that is not provided.
Clearly distinguish facts, inference and assumptions.
Do not provide personalized financial advice.
Use concise institutional research language.
"""

    user_prompt = f"""
Ticker: {ticker}

Market data:
{json.dumps(context, ensure_ascii=False, default=str)}

Return a structured research memo.
"""

    url = (
        f"{config['base_url']}"
        "/chat/completions"
    )

    payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": 0.25,
    }

    headers = {
        "Authorization":
            f"Bearer {config['api_key']}",
        "Content-Type":
            "application/json",
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=45,
        )

        response.raise_for_status()

        data = response.json()

        return (
            data
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )

    except Exception:
        return None


def run_ai_brain(
    ticker: str,
    fundamentals: Dict[str, Any],
    technicals: Dict[str, Any],
    brain: Dict[str, Any],
    scenarios: Dict[str, float],
) -> Dict[str, Any]:

    context = {
        "fundamentals": fundamentals,
        "technicals": technicals,
        "quant_brain": brain,
        "valuation_scenarios": scenarios,
    }

    local = local_ai_memo(
        ticker,
        fundamentals,
        technicals,
        brain,
        scenarios,
    )

    llm = call_llm(
        ticker,
        context,
    )

    return {
        "local": local,
        "llm": llm,
        "mode": (
            "LLM COMMITTEE"
            if llm
            else "QUANT + LOCAL BRAIN"
        ),
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-icon">◈</div>
            <div>
                <div class="brand-title">Simon</div>
                <div class="brand-subtitle">
                    Investment Brain
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            color:#677082;
            font-size:10px;
            margin-bottom:20px;
        ">
            AI-NATIVE US EQUITY INTELLIGENCE · V{APP_VERSION}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-title">Research</div>',
        unsafe_allow_html=True,
    )

    ticker_input = st.text_input(
        "Ticker",
        value=st.session_state.ticker,
        label_visibility="collapsed",
        placeholder="AAPL / NVDA / MSFT",
    )

    ticker = (
        ticker_input.strip().upper()
        if ticker_input.strip()
        else DEFAULT_TICKER
    )

    st.session_state.ticker = ticker

    st.markdown(
        '<div class="sidebar-title">Watchlist</div>',
        unsafe_allow_html=True,
    )

    watch_cols = st.columns(2)

    for i, symbol in enumerate(
        WATCHLIST
    ):

        with watch_cols[i % 2]:

            if st.button(
                symbol,
                key=f"watch_{symbol}",
                use_container_width=True,
            ):
                st.session_state.ticker = symbol
                st.rerun()

    st.markdown(
        '<div class="sidebar-title">Market Window</div>',
        unsafe_allow_html=True,
    )

    period = st.selectbox(
        "Period",
        [
            "1mo",
            "3mo",
            "6mo",
            "1y",
            "2y",
            "5y",
            "10y",
        ],
        index=3,
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="sidebar-title">Risk Profile</div>',
        unsafe_allow_html=True,
    )

    risk_profile = st.selectbox(
        "Risk",
        [
            "Conservative",
            "Balanced",
            "Growth",
            "Aggressive",
        ],
        index=1,
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="sidebar-title">System</div>',
        unsafe_allow_html=True,
    )

    ai_cfg = ai_config()

    if ai_cfg["configured"]:

        st.markdown(
            """
            <div class="online"
                 style="width:100%;
                        justify-content:center;
                        margin-bottom:8px;">
                <span class="dot"></span>
                AI BRAIN · ONLINE
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div style="
                width:100%;
                text-align:center;
                padding:10px;
                border-radius:12px;
                background:rgba(244,201,93,.08);
                border:1px solid rgba(244,201,93,.15);
                color:#e5c464;
                font-size:10px;
                font-weight:700;
            ">
                AI BRAIN · LOCAL MODE
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Live market data is best-effort. "
        "Fallback mode keeps the terminal usable."
    )


# ============================================================
# LOAD DATA
# ============================================================

with st.spinner(
    f"Synchronizing {ticker} intelligence..."
):

    history, market_source, market_error = (
        fetch_history(
            ticker,
            period,
        )
    )

    fundamentals, fundamental_source, fundamental_error = (
        fetch_profile(
            ticker
        )
    )

    quote = fetch_quote(
        ticker,
        history,
    )

    technical_df = calculate_technicals(
        history
    )

    technicals = technical_snapshot(
        technical_df
    )

    brain = compute_brain(
        ticker,
        fundamentals,
        technicals,
    )

    scenarios = valuation_scenarios(
        quote.get("price"),
        fundamentals,
    )


# ============================================================
# HEADER
# ============================================================

now = datetime.now().strftime(
    "%Y-%m-%d %H:%M"
)

data_is_live = (
    market_source == "LIVE"
)

data_badge_class = (
    "data-badge-live"
    if data_is_live
    else "data-badge-demo"
)

data_badge_text = (
    "MARKET ENGINE · LIVE"
    if data_is_live
    else "MARKET ENGINE · FALLBACK"
)

st.markdown(
    f"""
    <div class="topbar">

        <div class="topbar-left">
            US EQUITY INTELLIGENCE TERMINAL
            &nbsp;·&nbsp;
            {now}
        </div>

        <div style="
            display:flex;
            align-items:center;
            gap:8px;
        ">

            <span class="data-badge {data_badge_class}">
                {data_badge_text}
            </span>

            <span class="online">
                <span class="dot"></span>
                BRAIN READY
            </span>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FALLBACK NOTICE
# ============================================================

if not data_is_live:

    st.warning(
        "Live market data is temporarily unavailable. "
        "Simon Investment Brain has switched to fallback mode "
        "so the terminal remains usable. "
        f"Source detail: {market_error or 'unknown'}"
    )


# ============================================================
# HERO
# ============================================================

company_name = (
    fundamentals.get(
        "name"
    )
    or ticker
)

price = quote.get(
    "price"
)

change = quote.get(
    "change"
)

change_pct = quote.get(
    "change_percent"
)

if change is None:

    change_html = "—"

    change_class = "hero-change-flat"

else:

    if change > 0:
        change_class = "hero-change-up"
        sign = "+"
    elif change < 0:
        change_class = "hero-change-down"
        sign = ""
    else:
        change_class = "hero-change-flat"
        sign = ""

    change_html = (
        f"{sign}{fmt_money(change)}"
        f"&nbsp;&nbsp;"
        f"{sign}{fmt_percent(change_pct)}"
    )


st.markdown(
    f"""
    <div class="hero">

        <div class="hero-kicker">
            {fundamentals.get('sector') or 'US EQUITY'}
            &nbsp;·&nbsp;
            {fundamentals.get('industry') or 'MARKET'}
        </div>

        <div class="hero-symbol">
            {ticker}
        </div>

        <div class="hero-name">
            {company_name}
        </div>

        <div class="hero-price">
            {fmt_money(price)}
        </div>

        <div class="{change_class}"
             style="
                margin-top:10px;
                font-size:15px;
                font-weight:700;
             ">
            {change_html}
        </div>

        <div class="hero-meta">

            <span class="pill">
                Trend · {technicals.get('trend', '—')}
            </span>

            <span class="pill">
                Momentum · {technicals.get('momentum', '—')}
            </span>

            <span class="pill">
                AI · {brain['decision']}
            </span>

            <span class="pill">
                Risk · {brain['risk']}/100
            </span>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONVICTION + HERO METRICS
# ============================================================

st.write("")

left, right = st.columns(
    [1.15, 3.85]
)

with left:

    score = brain["conviction"]

    st.markdown(
        f"""
        <div class="conviction">

            <div class="conviction-label">
                AI CONVICTION
            </div>

            <div class="conviction-score">
                {score}
                <span style="
                    font-size:18px;
                    color:#747d8d;
                    font-weight:500;
                ">
                    /100
                </span>
            </div>

            <div class="conviction-bar">
                <div class="conviction-fill"
                     style="width:{score}%;">
                </div>
            </div>

            <div class="conviction-label-row">
                <span>LOW</span>
                <span>NEUTRAL</span>
                <span>HIGH</span>
            </div>

            <div style="
                margin-top:20px;
                font-size:12px;
                color:#aeb6c5;
            ">
                {brain['decision']}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with right:

    metrics = [
        (
            "MARKET CAP",
            fmt_large(
                fundamentals.get(
                    "market_cap"
                )
            ),
            "Equity value",
        ),
        (
            "P / E",
            (
                f"{safe_float(fundamentals.get('pe')):.1f}x"
                if safe_float(
                    fundamentals.get("pe")
                ) is not None
                else "—"
            ),
            "Trailing valuation",
        ),
        (
            "REVENUE GROWTH",
            fmt_percent(
                fundamentals.get(
                    "revenue_growth"
                )
            ),
            "YoY growth",
        ),
        (
            "ROE",
            fmt_percent(
                fundamentals.get(
                    "roe"
                )
            ),
            "Capital efficiency",
        ),
        (
            "RSI",
            (
                f"{technicals.get('rsi'):.1f}"
                if technicals.get(
                    "rsi"
                ) is not None
                else "—"
            ),
            "14D momentum",
        ),
    ]

    cols = st.columns(5)

    for col, item in zip(
        cols,
        metrics,
    ):

        label, value, caption = item

        with col:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        {label}
                    </div>

                    <div class="metric-value">
                        {value}
                    </div>

                    <div class="metric-caption">
                        {caption}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# MAIN NAV
# ============================================================

tabs = st.tabs(
    [
        "◉ Overview",
        "✦ AI Brain",
        "▦ Fundamentals",
        "⌁ Technicals",
        "△ Risk",
        "◎ Research",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    left, right = st.columns(
        [2.25, 1]
    )

    with left:

        st.markdown(
            """
            <div class="section-head">
                <div>
                    <div class="section-title">
                        Price Intelligence
                    </div>
                    <div class="section-subtitle">
                        MARKET STRUCTURE · TREND · MOMENTUM
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chart_cols = [
            "Close",
            "SMA20",
            "SMA50",
            "SMA200",
        ]

        chart_df = technical_df[
            [
                c for c in chart_cols
                if c in technical_df.columns
            ]
        ].dropna(
            how="all"
        )

        st.line_chart(
            chart_df,
            height=410,
        )

    with right:

        st.markdown(
            """
            <div class="section-head">
                <div>
                    <div class="section-title">
                        Market Signals
                    </div>
                    <div class="section-subtitle">
                        SYSTEMATIC SIGNALS
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        signal_items = [
            (
                "TREND REGIME",
                technicals.get(
                    "trend",
                    "UNKNOWN",
                ),
            ),
            (
                "MOMENTUM",
                technicals.get(
                    "momentum",
                    "UNKNOWN",
                ),
            ),
            (
                "MACD",
                technicals.get(
                    "macd_state",
                    "UNKNOWN",
                ),
            ),
            (
                "VOLATILITY",
                fmt_percent(
                    technicals.get(
                        "volatility_30d"
                    )
                ),
            ),
        ]

        for label, value in signal_items:

            st.markdown(
                f"""
                <div class="risk-card"
                     style="margin-bottom:10px;">

                    <div class="metric-label">
                        {label}
                    </div>

                    <div style="
                        margin-top:9px;
                        font-size:15px;
                        font-weight:800;
                    ">
                        {value}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # INVESTMENT THESIS
    # --------------------------------------------------------

    local = local_ai_memo(
        ticker,
        fundamentals,
        technicals,
        brain,
        scenarios,
    )

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">
                    Investment Thesis
                </div>
                <div class="section-subtitle">
                    AI-SYNTHESIZED DECISION FRAMEWORK
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="thesis-card">

            <div class="thesis-title">
                {brain['decision']}
            </div>

            <div class="thesis-text">
                {local['thesis']}
            </div>

            <div style="
                height:1px;
                background:rgba(255,255,255,.06);
                margin:18px 0;
            "></div>

            <div class="thesis-text">
                {local['quality']}
                <br><br>
                {local['growth']}
                <br><br>
                {local['valuation']}
                <br><br>
                {local['technical']}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# AI BRAIN
# ============================================================

with tabs[1]:

    st.markdown(
        """
        <div class="brain">

            <div class="brain-header">

                <div>
                    <div class="brain-title">
                        Investment Brain
                    </div>

                    <div class="section-subtitle">
                        VALUE · QUALITY · GROWTH · MOMENTUM · RISK
                    </div>
                </div>

                <div class="brain-status">
                    ● COMPUTING
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    agents = [
        (
            "VALUE ENGINE",
            brain["valuation"],
            "Intrinsic value & valuation discipline",
        ),
        (
            "BUSINESS QUALITY",
            brain["quality"],
            "Moat · margins · capital efficiency",
        ),
        (
            "GROWTH ENGINE",
            brain["growth"],
            "Revenue · earnings · structural growth",
        ),
        (
            "MOMENTUM ENGINE",
            brain["momentum"],
            "Trend · RSI · market structure",
        ),
        (
            "BALANCE SHEET",
            brain["balance"],
            "Leverage · liquidity · resilience",
        ),
    ]

    agent_cols = st.columns(5)

    for col, item in zip(
        agent_cols,
        agents,
    ):

        name, score_value, note = item

        with col:

            st.markdown(
                f"""
                <div class="agent">

                    <div class="agent-name">
                        {name}
                    </div>

                    <div class="agent-score">
                        {score_value}
                    </div>

                    <div class="agent-line">
                        <div class="agent-fill"
                             style="width:{score_value}%;">
                        </div>
                    </div>

                    <div class="agent-note">
                        {note}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    # --------------------------------------------------------
    # VALUATION
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">
                    Scenario Engine
                </div>
                <div class="section-subtitle">
                    BEAR · BASE · BULL
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scenario_cols = st.columns(3)

    scenario_data = [
        (
            scenario_cols[0],
            "BEAR CASE",
            scenarios["bear"],
            "scenario-down",
        ),
        (
            scenario_cols[1],
            "BASE CASE",
            scenarios["base"],
            "scenario-neutral",
        ),
        (
            scenario_cols[2],
            "BULL CASE",
            scenarios["bull"],
            "scenario-up",
        ),
    ]

    for col, label, value, css_class in scenario_data:

        with col:

            upside = (
                value / price - 1
                if price
                else None
            )

            st.markdown(
                f"""
                <div class="scenario">

                    <div class="scenario-name">
                        {label}
                    </div>

                    <div class="scenario-price {css_class}">
                        {fmt_money(value)}
                    </div>

                    <div style="
                        margin-top:6px;
                        color:#747d8e;
                        font-size:10px;
                    ">
                        {fmt_percent(upside)}
                        vs current
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    # --------------------------------------------------------
    # RUN RESEARCH
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">
                    Research Committee
                </div>
                <div class="section-subtitle">
                    LOCAL QUANT BRAIN + OPTIONAL LLM COMMITTEE
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "✦ Run Investment Committee",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Investment Brain is reasoning across multiple layers..."
        ):

            st.session_state.research = run_ai_brain(
                ticker,
                fundamentals,
                technicals,
                brain,
                scenarios,
            )

    research = st.session_state.research

    if research is None:

        st.info(
            "Click Run Investment Committee to generate a full research memo."
        )

    else:

        local = research["local"]

        st.markdown(
            f"""
            <div class="brain">

                <div class="brain-header">

                    <div>
                        <div class="brain-title">
                            {research['mode']}
                        </div>

                        <div class="section-subtitle">
                            {ticker} · INVESTMENT COMMITTEE OUTPUT
                        </div>
                    </div>

                    <div class="brain-status">
                        CONVICTION {brain['conviction']}/100
                    </div>

                </div>

                <div style="
                    margin-top:22px;
                    color:#c1c8d5;
                    font-size:13px;
                    line-height:1.8;
                    white-space:pre-wrap;
                ">
                    {research['llm'] or local['committee']}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# FUNDAMENTALS
# ============================================================

with tabs[2]:

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">
                    Fundamental Intelligence
                </div>
                <div class="section-subtitle">
                    BUSINESS QUALITY · GROWTH · VALUATION · BALANCE SHEET
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4 = st.columns(4)

    fundamental_blocks = [
        (
            f1,
            "VALUATION",
            [
                (
                    "P / E",
                    (
                        f"{safe_float(fundamentals.get('pe')):.1f}x"
                        if safe_float(
                            fundamentals.get("pe")
                        ) is not None
                        else "—"
                    ),
                ),
                (
                    "Forward P/E",
                    (
                        f"{safe_float(fundamentals.get('forward_pe')):.1f}x"
                        if safe_float(
                            fundamentals.get("forward_pe")
                        ) is not None
                        else "—"
                    ),
                ),
                (
                    "P / B",
                    (
                        f"{safe_float(fundamentals.get('price_to_book')):.1f}x"
                        if safe_float(
                            fundamentals.get("price_to_book")
                        ) is not None
                        else "—"
                    ),
                ),
            ],
        ),
        (
            f2,
            "GROWTH",
            [
                (
                    "Revenue Growth",
                    fmt_percent(
                        fundamentals.get(
                            "revenue_growth"
                        )
                    ),
                ),
                (
                    "Earnings Growth",
                    fmt_percent(
                        fundamentals.get(
                            "earnings_growth"
                        )
                    ),
                ),
                (
                    "ROE",
                    fmt_percent(
                        fundamentals.get(
                            "roe"
                        )
                    ),
                ),
            ],
        ),
        (
            f3,
            "PROFITABILITY",
            [
                (
                    "Gross Margin",
                    fmt_percent(
                        fundamentals.get(
                            "gross_margin"
                        )
                    ),
                ),
                (
                    "Operating Margin",
                    fmt_percent(
                        fundamentals.get(
                            "operating_margin"
                        )
                    ),
                ),
                (
                    "Free Cash Flow",
                    fmt_large(
                        fundamentals.get(
                            "free_cash_flow"
                        )
                    ),
                ),
            ],
        ),
        (
            f4,
            "BALANCE SHEET",
            [
                (
                    "Cash",
                    fmt_large(
                        fundamentals.get(
                            "cash"
                        )
                    ),
                ),
                (
                    "Debt",
                    fmt_large(
                        fundamentals.get(
                            "debt"
                        )
                    ),
                ),
                (
                    "Debt / Equity",
                    (
                        f"{safe_float(fundamentals.get('debt_to_equity')):.1f}%"
                        if safe_float(
                            fundamentals.get(
                                "debt_to_equity"
                            )
                        ) is not None
                        else "—"
                    ),
                ),
            ],
        ),
    ]

    for col, title, rows in fundamental_blocks:

        with col:

            html = f"""
            <div class="thesis-card">

                <div class="metric-label">
                    {title}
                </div>
            """

            for label, value in rows:

                html += f"""
                    <div style="
                        display:flex;
                        justify-content:space-between;
                        margin-top:17px;
                        font-size:11px;
                    ">
                        <span style="color:#788294;">
                            {label}
                        </span>

                        <span style="
                            color:#e4e8ef;
                            font-weight:700;
                        ">
                            {value}
                        </span>
                    </div>
                """

            html += "</div>"

            st.markdown(
                html,
                unsafe_allow_html=True,
            )

    st.write("")

    st.markdown(
        f"""
        <div class="glass"
             style="padding:22px;">

            <div class="metric-label">
                COMPANY PROFILE
            </div>

            <div style="
                margin-top:16px;
                display:grid;
                grid-template-columns:
                    repeat(2, minmax(0,1fr));
                gap:16px;
                color:#9da6b5;
                font-size:12px;
            ">

                <div>
                    <b style="color:#e2e6ee;">
                        Company
                    </b><br>
                    {company_name}
                </div>

                <div>
                    <b style="color:#e2e6ee;">
                        Sector
                    </b><br>
                    {fundamentals.get('sector') or '—'}
                </div>

                <div>
                    <b style="color:#e2e6ee;">
                        Industry
                    </b><br>
                    {fundamentals.get('industry') or '—'}
                </div>

                <div>
                    <b style="color:#e2e6ee;">
                        Country
                    </b><br>
                    {fundamentals.get('country') or '—'}
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TECHNICALS
# ============================================================

with tabs[3]:

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">
                    Technical Intelligence
                </div>
                <div class="section-subtitle">
                    TREND · MOVING AVERAGES · RSI · MACD
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    technical_chart = technical_df[
        [
            c for c in [
                "Close",
                "SMA20",
                "SMA50",
                "SMA200",
            ]
            if c in technical_df.columns
        ]
    ]

    st.line_chart(
        technical_chart,
        height=460,
    )

    tcols = st.columns(5)

    technical_metrics = [
        (
            "PRICE",
            fmt_money(
                technicals.get("price")
            ),
        ),
        (
            "SMA 20",
            fmt_money(
                technicals.get("sma20")
            ),
        ),
        (
            "SMA 50",
            fmt_money(
                technicals.get("sma50")
            ),
        ),
        (
            "SMA 200",
            fmt_money(
                technicals.get("sma200")
            ),
        ),
        (
            "RSI",
            (
                f"{technicals.get('rsi'):.2f}"
                if technicals.get("rsi")
                is not None
                else "—"
            ),
        ),
    ]

    for col, (label, value) in zip(
        tcols,
        technical_metrics,
    ):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        {label}
                    </div>

                    <div class="metric-value">
                        {value}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# RISK
# ============================================================

with tabs[4]:

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">
                    Risk Intelligence
                </div>
                <div class="section-subtitle">
                    DOWNSIDE · VOLATILITY · BALANCE SHEET · THESIS RISK
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    r1, r2, r3 = st.columns(3)

    risk_items = [
        (
            r1,
            "OVERALL RISK",
            brain["risk"],
        ),
        (
            r2,
            "BALANCE SHEET",
            100 - brain["balance"],
        ),
        (
            r3,
            "VALUATION RISK",
            100 - brain["valuation"],
        ),
    ]

    for col, label, value in risk_items:

        with col:

            css = score_class(
                100 - value
            )

            st.markdown(
                f"""
                <div class="risk-card">

                    <div class="metric-label">
                        {label}
                    </div>

                    <div class="risk-value {css}"
                         style="margin-top:10px;">
                        {int(value)}/100
                    </div>

                    <div style="
                        margin-top:10px;
                        height:6px;
                        background:rgba(255,255,255,.06);
                        border-radius:999px;
                    ">
                        <div style="
                            width:{int(value)}%;
                            height:100%;
                            border-radius:999px;
                            background:linear-gradient(
                                90deg,
                                #4fd6a0,
                                #f1c95d,
                                #ff6375
                            );
                        "></div>
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    local = local_ai_memo(
        ticker,
        fundamentals,
        technicals,
        brain,
        scenarios,
    )

    st.markdown(
        f"""
        <div class="thesis-card">

            <div class="metric-label">
                RISK COMMITTEE
            </div>

            <div class="thesis-text"
                 style="margin-top:14px;">
                {local['risk']}
            </div>

            <div style="
                height:1px;
                background:rgba(255,255,255,.06);
                margin:18px 0;
            "></div>

            <div class="metric-label">
                WHAT CAN BREAK THE THESIS
            </div>

            <div class="thesis-text"
                 style="margin-top:14px;">
                • Earnings expectations fall sharply<br>
                • Valuation multiple compresses<br>
                • Growth decelerates faster than expected<br>
                • Macro liquidity deteriorates<br>
                • Company-specific competitive pressure increases
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RESEARCH
# ============================================================

with tabs[5]:

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">
                    Research Terminal
                </div>
                <div class="section-subtitle">
                    INSTITUTIONAL-STYLE INVESTMENT MEMO
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    local = local_ai_memo(
        ticker,
        fundamentals,
        technicals,
        brain,
        scenarios,
    )

    c1, c2 = st.columns(
        [1, 2]
    )

    with c1:

        st.markdown(
            f"""
            <div class="brain">

                <div class="metric-label">
                    RECOMMENDATION
                </div>

                <div style="
                    font-size:28px;
                    font-weight:850;
                    margin-top:12px;
                ">
                    {brain['decision']}
                </div>

                <div style="
                    margin-top:20px;
                    color:#8992a3;
                    font-size:11px;
                    line-height:1.7;
                ">
                    Conviction:
                    <b style="color:#e8ecf4;">
                        {brain['conviction']}/100
                    </b>
                    <br>
                    Risk:
                    <b style="color:#e8ecf4;">
                        {brain['risk']}/100
                    </b>
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
            <div class="thesis-card">

                <div class="metric-label">
                    CORE THESIS
                </div>

                <div class="thesis-text"
                     style="margin-top:14px;">
                    {local['thesis']}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    sections = [
        (
            "01 · BUSINESS QUALITY",
            local["quality"],
        ),
        (
            "02 · GROWTH",
            local["growth"],
        ),
        (
            "03 · VALUATION",
            local["valuation"],
        ),
        (
            "04 · TECHNICAL STRUCTURE",
            local["technical"],
        ),
        (
            "05 · RISK",
            local["risk"],
        ),
    ]

    for title, text in sections:

        st.markdown(
            f"""
            <div class="glass-soft"
                 style="padding:18px;
                        margin-bottom:10px;">

                <div class="metric-label">
                    {title}
                </div>

                <div style="
                    margin-top:9px;
                    color:#a5adbc;
                    font-size:12px;
                    line-height:1.7;
                ">
                    {text}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer">

        <b style="color:#727b8b;">
            {APP_NAME} V{APP_VERSION}
        </b>

        &nbsp;·&nbsp;

        AI-native US equity research terminal

        <br>

        Data:
        {market_source}
        &nbsp;·&nbsp;
        Fundamentals:
        {fundamental_source}
        &nbsp;·&nbsp;
        AI:
        {
            "LLM + Quant"
            if ai_cfg["configured"]
            else "Local Quant Brain"
        }

        <br><br>

        Research tool only.
        AI scores, valuation scenarios and research outputs
        are analytical frameworks, not financial advice.

    </div>
    """,
    unsafe_allow_html=True,
)