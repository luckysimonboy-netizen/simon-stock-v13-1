from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# OPTIONAL BACKEND IMPORTS
# ============================================================

try:
    from data.market import (
        get_market_bundle,
        get_quote,
        get_fundamental_snapshot,
        get_history,
        data_health_check,
    )

    MARKET_BACKEND = True

except Exception:
    MARKET_BACKEND = False

    get_market_bundle = None
    get_quote = None
    get_fundamental_snapshot = None
    get_history = None
    data_health_check = None


try:
    from ai.orchestrator import (
        ai_health_check,
        run_full_ai_research,
    )

    AI_BACKEND = True

except Exception:
    AI_BACKEND = False

    ai_health_check = None
    run_full_ai_research = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SIMON Investment Brain",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "ticker": "NVDA",
    "period": "1y",
    "ai_report": None,
    "research_running": False,
    "last_refresh": None,
    "watchlist": [
        "NVDA",
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
    ],
    "density": "Comfortable",
    "theme": "System",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# LIQUID GLASS CSS
# ============================================================

st.markdown(
    """
<style>

/* ============================================================
   ROOT
   ============================================================ */

:root {
    --glass-border: rgba(255,255,255,.13);
    --glass-bg: rgba(255,255,255,.055);
    --glass-bg-strong: rgba(255,255,255,.085);
    --muted: rgba(255,255,255,.55);
    --text: rgba(255,255,255,.94);
    --blue: #6f8cff;
    --cyan: #70d7ff;
    --green: #50d890;
    --red: #ff667c;
    --radius: 24px;
}

.stApp {
    background:
        radial-gradient(
            circle at 5% 0%,
            rgba(83,112,255,.17),
            transparent 28%
        ),
        radial-gradient(
            circle at 92% 4%,
            rgba(75,201,255,.12),
            transparent 24%
        ),
        radial-gradient(
            circle at 55% 100%,
            rgba(100,75,255,.08),
            transparent 30%
        ),
        #080a10;
    color: var(--text);
}

/* ============================================================
   REMOVE DEFAULT STREAMLIT SPACE
   ============================================================ */

.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 1700px;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            rgba(16,19,28,.94),
            rgba(8,10,16,.97)
        );
    border-right: 1px solid rgba(255,255,255,.07);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

/* ============================================================
   GLOBAL BUTTON
   ============================================================ */

.stButton > button {
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,.10);
    background: rgba(255,255,255,.065);
    color: rgba(255,255,255,.92);
    transition:
        transform .18s ease,
        background .18s ease,
        border .18s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    background: rgba(255,255,255,.105);
    border-color: rgba(255,255,255,.20);
}

.stButton > button[kind="primary"] {
    background:
        linear-gradient(
            135deg,
            rgba(105,130,255,.90),
            rgba(87,182,255,.80)
        );
    border: 1px solid rgba(255,255,255,.18);
}

/* ============================================================
   HEADER
   ============================================================ */

.sb-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
}

.sb-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.sb-logo {
    width: 44px;
    height: 44px;
    border-radius: 15px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 21px;
    font-weight: 900;

    background:
        linear-gradient(
            135deg,
            rgba(120,145,255,.85),
            rgba(65,201,255,.55)
        );

    box-shadow:
        0 10px 35px rgba(78,122,255,.20),
        inset 0 1px 0 rgba(255,255,255,.35);
}

.sb-title {
    font-size: 24px;
    font-weight: 850;
    letter-spacing: -1px;
}

.sb-subtitle {
    font-size: 11px;
    color: var(--muted);
    margin-top: 2px;
}

.sb-status {
    display: flex;
    align-items: center;
    gap: 7px;

    padding: 8px 12px;
    border-radius: 999px;

    background: rgba(80,216,144,.08);
    border: 1px solid rgba(80,216,144,.14);

    font-size: 11px;
    color: rgba(220,255,235,.86);
}

.sb-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #50d890;
    box-shadow: 0 0 12px rgba(80,216,144,.8);
}

/* ============================================================
   GLASS CARD
   ============================================================ */

.glass {
    border-radius: var(--radius);

    border: 1px solid var(--glass-border);

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,.075),
            rgba(255,255,255,.025)
        );

    backdrop-filter: blur(30px);
    -webkit-backdrop-filter: blur(30px);

    box-shadow:
        0 18px 60px rgba(0,0,0,.18),
        inset 0 1px 0 rgba(255,255,255,.055);

    padding: 21px;

    margin-bottom: 15px;
}

/* ============================================================
   HERO
   ============================================================ */

.hero {
    position: relative;
    overflow: hidden;

    border-radius: 30px;

    border: 1px solid rgba(150,170,255,.16);

    background:
        radial-gradient(
            circle at 90% 20%,
            rgba(83,116,255,.20),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            rgba(255,255,255,.095),
            rgba(255,255,255,.025)
        );

    padding: 30px;

    margin-bottom: 15px;

    backdrop-filter: blur(35px);
    -webkit-backdrop-filter: blur(35px);

    box-shadow:
        0 25px 90px rgba(0,0,0,.20),
        inset 0 1px 0 rgba(255,255,255,.08);
}

.hero::after {
    content: "";
    position: absolute;

    width: 260px;
    height: 260px;

    right: -100px;
    top: -100px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(104,139,255,.20),
            transparent 70%
        );

    pointer-events: none;
}

.hero-symbol {
    font-size: 12px;
    letter-spacing: 2px;
    color: rgba(255,255,255,.50);
    font-weight: 750;
}

.hero-company {
    font-size: 31px;
    font-weight: 850;
    letter-spacing: -1.3px;
    margin-top: 4px;
}

.hero-price {
    font-size: 47px;
    font-weight: 900;
    letter-spacing: -2.5px;
    margin-top: 12px;
}

.hero-change {
    display: inline-flex;
    align-items: center;

    margin-top: 7px;

    padding: 7px 11px;

    border-radius: 999px;

    font-size: 13px;
    font-weight: 700;
}

.hero-meta {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 17px;
}

.meta-pill {
    padding: 6px 10px;
    border-radius: 999px;

    background: rgba(255,255,255,.055);
    border: 1px solid rgba(255,255,255,.075);

    font-size: 11px;
    color: rgba(255,255,255,.63);
}

/* ============================================================
   KPI
   ============================================================ */

.kpi {
    min-height: 118px;
}

.kpi-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    color: rgba(255,255,255,.43);
}

.kpi-value {
    font-size: 25px;
    font-weight: 850;
    margin-top: 8px;
    letter-spacing: -.7px;
}

.kpi-sub {
    font-size: 11px;
    color: rgba(255,255,255,.42);
    margin-top: 5px;
}

/* ============================================================
   SECTION
   ============================================================ */

.section-title {
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -.4px;
    margin: 17px 0 10px 1px;
}

.section-sub {
    color: rgba(255,255,255,.42);
    font-size: 11px;
    margin-top: -7px;
    margin-bottom: 11px;
}

/* ============================================================
   BRAIN
   ============================================================ */

.brain-card {
    border-radius: 27px;

    border: 1px solid rgba(121,146,255,.20);

    background:
        radial-gradient(
            circle at 15% 0%,
            rgba(100,126,255,.15),
            transparent 32%
        ),
        linear-gradient(
            135deg,
            rgba(91,117,255,.095),
            rgba(255,255,255,.025)
        );

    padding: 23px;

    backdrop-filter: blur(32px);
    -webkit-backdrop-filter: blur(32px);

    box-shadow:
        0 20px 70px rgba(52,74,180,.13),
        inset 0 1px 0 rgba(255,255,255,.07);
}

.brain-label {
    font-size: 10px;
    letter-spacing: 1.7px;
    color: rgba(150,171,255,.72);
    text-transform: uppercase;
}

.brain-title {
    font-size: 23px;
    font-weight: 850;
    margin-top: 3px;
}

.brain-description {
    font-size: 12px;
    line-height: 1.65;
    color: rgba(255,255,255,.55);
    max-width: 800px;
    margin-top: 7px;
}

/* ============================================================
   SCORE
   ============================================================ */

.score-box {
    text-align: center;
    padding: 15px;
}

.score-number {
    font-size: 47px;
    font-weight: 900;
    letter-spacing: -2px;
}

.score-caption {
    font-size: 10px;
    color: rgba(255,255,255,.45);
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ============================================================
   AGENT
   ============================================================ */

.agent {
    border-radius: 18px;
    padding: 15px;

    background: rgba(255,255,255,.038);
    border: 1px solid rgba(255,255,255,.065);

    min-height: 110px;
}

.agent-name {
    font-size: 10px;
    letter-spacing: 1px;
    color: rgba(255,255,255,.45);
    text-transform: uppercase;
}

.agent-score {
    font-size: 25px;
    font-weight: 850;
    margin-top: 5px;
}

.agent-note {
    font-size: 10px;
    color: rgba(255,255,255,.43);
    margin-top: 3px;
}

/* ============================================================
   WATCHLIST
   ============================================================ */

.watch-item {
    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 10px 11px;
    margin: 5px 0;

    border-radius: 13px;

    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.05);
}

.watch-symbol {
    font-weight: 800;
    font-size: 12px;
}

.watch-price {
    font-size: 11px;
    color: rgba(255,255,255,.54);
}

.up {
    color: var(--green);
}

.down {
    color: var(--red);
}

.neutral {
    color: rgba(255,255,255,.60);
}

/* ============================================================
   SIGNAL
   ============================================================ */

.signal {
    border-radius: 17px;
    padding: 15px;
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.07);
}

.signal-label {
    font-size: 9px;
    letter-spacing: 1px;
    color: rgba(255,255,255,.42);
    text-transform: uppercase;
}

.signal-value {
    font-size: 17px;
    font-weight: 800;
    margin-top: 5px;
}

/* ============================================================
   RESEARCH
   ============================================================ */

.research {
    border-left: 2px solid rgba(111,140,255,.65);
    padding-left: 15px;
    line-height: 1.75;
    color: rgba(255,255,255,.68);
    font-size: 13px;
}

/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;
    color: rgba(255,255,255,.25);
    font-size: 10px;
    padding: 20px;
}

/* ============================================================
   STREAMLIT TABS
   ============================================================ */

button[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 13px !important;
    font-size: 12px !important;
}

div[data-baseweb="tab-list"] {
    gap: 5px;
}

/* ============================================================
   INPUT
   ============================================================ */

div[data-baseweb="input"] {
    border-radius: 14px;
}

/* ============================================================
   METRIC
   ============================================================ */

div[data-testid="stMetric"] {
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 18px;
    padding: 14px;
}

/* ============================================================
   DATAFRAME
   ============================================================ */

div[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 900px) {

    .hero-price {
        font-size: 38px;
    }

    .hero-company {
        font-size: 25px;
    }

    .sb-status {
        display: none;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None

        result = float(value)

        if math.isnan(result) or math.isinf(result):
            return None

        return result

    except Exception:
        return None


def fmt_money(value: Any, digits: int = 2) -> str:
    value = safe_float(value)

    if value is None:
        return "—"

    return f"${value:,.{digits}f}"


def fmt_percent(value: Any, digits: int = 2) -> str:
    value = safe_float(value)

    if value is None:
        return "—"

    # Most backend percentages are represented as decimals.
    if abs(value) <= 2:
        value *= 100

    return f"{value:.{digits}f}%"


def fmt_number(value: Any) -> str:
    value = safe_float(value)

    if value is None:
        return "—"

    absolute = abs(value)

    if absolute >= 1e12:
        return f"{value / 1e12:.2f}T"

    if absolute >= 1e9:
        return f"{value / 1e9:.2f}B"

    if absolute >= 1e6:
        return f"{value / 1e6:.2f}M"

    if absolute >= 1e3:
        return f"{value / 1e3:.2f}K"

    return f"{value:,.0f}"


def get_value(data: Dict, *keys, default=None):
    if not isinstance(data, dict):
        return default

    for key in keys:
        value = data.get(key)

        if value is not None:
            return value

    return default


def safe_dict(value):
    return value if isinstance(value, dict) else {}


def calculate_technicals(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    # Normalize columns
    rename_map = {}

    for col in data.columns:

        normalized = str(col).lower()

        if normalized == "close":
            rename_map[col] = "Close"

        elif normalized == "open":
            rename_map[col] = "Open"

        elif normalized == "high":
            rename_map[col] = "High"

        elif normalized == "low":
            rename_map[col] = "Low"

        elif normalized == "volume":
            rename_map[col] = "Volume"

    data = data.rename(columns=rename_map)

    if "Close" not in data.columns:
        return data

    close = pd.to_numeric(
        data["Close"],
        errors="coerce",
    )

    data["SMA20"] = close.rolling(20).mean()
    data["SMA50"] = close.rolling(50).mean()
    data["SMA200"] = close.rolling(200).mean()

    data["EMA12"] = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    data["EMA26"] = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    data["MACD"] = (
        data["EMA12"] -
        data["EMA26"]
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

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    data["RSI"] = 100 - (
        100 / (1 + rs)
    )

    if "Volume" in data.columns:

        volume = pd.to_numeric(
            data["Volume"],
            errors="coerce",
        )

        data["Volume_MA20"] = (
            volume.rolling(20).mean()
        )

    return data


def technical_summary(df):

    if df is None or df.empty:
        return {
            "trend": "UNKNOWN",
            "momentum": "UNKNOWN",
            "rsi": None,
            "sma20": None,
            "sma50": None,
            "sma200": None,
        }

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

    return {
        "trend": trend,
        "momentum": momentum,
        "rsi": rsi,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
    }


def calculate_brain_scores(
    fundamentals: Dict,
    technicals: Dict,
):

    f = safe_dict(fundamentals)

    revenue_growth = safe_float(
        get_value(
            f,
            "revenue_growth",
            "revenueGrowth",
        )
    )

    earnings_growth = safe_float(
        get_value(
            f,
            "earnings_growth",
            "earningsGrowth",
        )
    )

    roe = safe_float(
        get_value(
            f,
            "roe",
            "return_on_equity",
        )
    )

    margin = safe_float(
        get_value(
            f,
            "operating_margin",
            "operatingMargin",
        )
    )

    pe = safe_float(
        get_value(
            f,
            "pe",
            "trailing_pe",
        )
    )

    debt = safe_float(
        get_value(
            f,
            "debt_to_equity",
            "debtToEquity",
        )
    )

    # Normalize
    if revenue_growth is not None and abs(revenue_growth) <= 2:
        revenue_growth *= 100

    if earnings_growth is not None and abs(earnings_growth) <= 2:
        earnings_growth *= 100

    if roe is not None and abs(roe) <= 2:
        roe *= 100

    if margin is not None and abs(margin) <= 2:
        margin *= 100

    # QUALITY
    quality = 55

    if roe is not None:
        quality += min(max(roe / 3, -15), 20)

    if margin is not None:
        quality += min(max(margin / 2, -10), 15)

    quality = max(0, min(100, quality))

    # GROWTH
    growth = 50

    if revenue_growth is not None:
        growth += min(max(revenue_growth * 0.6, -25), 30)

    if earnings_growth is not None:
        growth += min(max(earnings_growth * 0.4, -20), 25)

    growth = max(0, min(100, growth))

    # VALUATION
    valuation = 55

    if pe is not None:

        if pe < 15:
            valuation += 25

        elif pe < 22:
            valuation += 12

        elif pe < 30:
            valuation += 2

        elif pe < 45:
            valuation -= 10

        else:
            valuation -= 20

    valuation = max(0, min(100, valuation))

    # MOMENTUM
    momentum = 50

    if technicals.get("trend") == "BULLISH":
        momentum += 28

    elif technicals.get("trend") == "BEARISH":
        momentum -= 28

    if technicals.get("momentum") == "POSITIVE":
        momentum += 12

    elif technicals.get("momentum") == "NEGATIVE":
        momentum -= 12

    elif technicals.get("momentum") == "OVERBOUGHT":
        momentum += 5

    elif technicals.get("momentum") == "OVERSOLD":
        momentum -= 3

    momentum = max(0, min(100, momentum))

    # MOAT
    moat = (
        quality * 0.55
        + growth * 0.30
        + valuation * 0.15
    )

    moat = max(0, min(100, moat))

    # RISK
    risk = 40

    if debt is not None:

        if debt > 200:
            risk += 30

        elif debt > 120:
            risk += 18

        elif debt > 60:
            risk += 8

        elif debt < 30:
            risk -= 8

    if technicals.get("trend") == "BEARISH":
        risk += 15

    if technicals.get("momentum") == "OVERBOUGHT":
        risk += 8

    risk = max(0, min(100, risk))

    conviction = (
        quality * 0.22
        + moat * 0.20
        + growth * 0.22
        + valuation * 0.16
        + momentum * 0.20
        - risk * 0.08
    )

    conviction = max(0, min(100, conviction))

    return {
        "quality": round(quality),
        "moat": round(moat),
        "growth": round(growth),
        "valuation": round(valuation),
        "momentum": round(momentum),
        "risk": round(risk),
        "conviction": round(conviction),
    }


# ============================================================
# FALLBACK DATA
# ============================================================

def fallback_quote(ticker):

    return {
        "price": None,
        "change": None,
        "change_percent": None,
        "volume": None,
    }


def fallback_fundamentals(ticker):

    return {
        "name": ticker,
        "sector": "—",
        "industry": "—",
        "country": "—",
        "market_cap": None,
        "pe": None,
        "forward_pe": None,
        "price_to_book": None,
        "ev_to_ebitda": None,
        "revenue_growth": None,
        "earnings_growth": None,
        "gross_margin": None,
        "operating_margin": None,
        "roe": None,
        "total_cash": None,
        "total_debt": None,
        "current_ratio": None,
        "debt_to_equity": None,
        "free_cash_flow": None,
        "website": None,
    }


# ============================================================
# LOAD MARKET DATA SAFELY
# ============================================================

@st.cache_data(
    ttl=60,
    show_spinner=False,
)
def load_market_data(
    ticker: str,
    period: str,
):

    result = {
        "history": pd.DataFrame(),
        "quote": fallback_quote(ticker),
        "fundamentals": fallback_fundamentals(ticker),
        "error": None,
    }

    if not MARKET_BACKEND:

        result["error"] = (
            "Market backend unavailable. "
            "Check data.market."
        )

        return result

    try:

        if get_market_bundle is not None:

            bundle = get_market_bundle(
                ticker,
                period=period,
                interval="1d",
            )

            if isinstance(bundle, dict):

                result["history"] = (
                    bundle.get(
                        "history",
                        pd.DataFrame(),
                    )
                    or pd.DataFrame()
                )

                result["quote"] = (
                    bundle.get(
                        "quote",
                        result["quote"],
                    )
                    or result["quote"]
                )

                result["fundamentals"] = (
                    bundle.get(
                        "fundamentals",
                        result["fundamentals"],
                    )
                    or result["fundamentals"]
                )

                return result

        # fallback individual loaders

        if get_history is not None:

            result["history"] = (
                get_history(
                    ticker,
                    period=period,
                    interval="1d",
                )
                or pd.DataFrame()
            )

        if get_quote is not None:

            result["quote"] = (
                get_quote(ticker)
                or result["quote"]
            )

        if get_fundamental_snapshot is not None:

            result["fundamentals"] = (
                get_fundamental_snapshot(ticker)
                or result["fundamentals"]
            )

    except Exception as exc:

        result["error"] = str(exc)

    return result


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:22px;
            font-weight:850;
            letter-spacing:-.7px;
        ">
        ◈ SIMON
        </div>
        <div style="
            font-size:10px;
            color:rgba(255,255,255,.42);
            letter-spacing:1.5px;
            margin-top:2px;
        ">
        INVESTMENT BRAIN · V14.0
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    ticker_input = st.text_input(
        "Search security",
        value=st.session_state.ticker,
        placeholder="NVDA / AAPL / MSFT",
        label_visibility="collapsed",
    )

    ticker = (
        ticker_input.strip().upper()
        if ticker_input.strip()
        else "NVDA"
    )

    st.session_state.ticker = ticker

    st.divider()

    st.markdown(
        "### Watchlist"
    )

    watchlist = st.session_state.watchlist

    for symbol in watchlist:

        if st.button(
            symbol,
            key=f"watch_{symbol}",
            use_container_width=True,
        ):

            st.session_state.ticker = symbol
            st.session_state.ai_report = None
            st.rerun()

    st.divider()

    st.markdown(
        "### Market Window"
    )

    period = st.selectbox(
        "History",
        [
            "1mo",
            "3mo",
            "6mo",
            "1y",
            "2y",
            "5y",
            "10y",
            "max",
        ],
        index=3,
        label_visibility="collapsed",
    )

    st.session_state.period = period

    st.divider()

    st.markdown(
        "### Terminal"
    )

    density = st.selectbox(
        "Information density",
        [
            "Compact",
            "Comfortable",
            "Research",
        ],
        index=1,
        label_visibility="collapsed",
    )

    st.session_state.density = density

    st.divider()

    st.markdown(
        "### System"

    )

    market_online = MARKET_BACKEND

    if market_online:

        st.success(
            "MARKET ENGINE · ONLINE"
        )

    else:

        st.error(
            "MARKET ENGINE · OFFLINE"
        )

    if AI_BACKEND:

        try:

            ai_status = ai_health_check()

        except Exception:

            ai_status = {
                "configured": False,
                "provider": "Unavailable",
                "model": "—",
            }

    else:

        ai_status = {
            "configured": False,
            "provider": "Unavailable",
            "model": "—",
        }

    if ai_status.get("configured"):

        st.success(
            "AI BRAIN · ONLINE"
        )

        st.caption(
            f"{ai_status.get('provider', 'AI')} · "
            f"{ai_status.get('model', '')}"
        )

    else:

        st.warning(
            "AI BRAIN · STANDBY"
        )

        st.caption(
            "Configure API credentials to enable deep research."
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="sb-header">

<div class="sb-brand">

<div class="sb-logo">
◈
</div>

<div>
<div class="sb-title">
SIMON Investment Brain
</div>

<div class="sb-subtitle">
AI-NATIVE US EQUITY INTELLIGENCE TERMINAL
</div>
</div>

</div>

<div class="sb-status">
<span class="sb-dot"></span>
MARKET ENGINE READY
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MARKET LOAD
# ============================================================

with st.spinner(
    f"Connecting to {ticker} market data..."
):

    market = load_market_data(
        ticker,
        period,
    )


history = market["history"]
quote = safe_dict(
    market["quote"]
)
fundamentals = safe_dict(
    market["fundamentals"]
)

if history is None:
    history = pd.DataFrame()

technical_df = calculate_technicals(
    history
)

technicals = technical_summary(
    technical_df
)

brain = calculate_brain_scores(
    fundamentals,
    technicals,
)


# ============================================================
# NORMALIZE QUOTE
# ============================================================

price = get_value(
    quote,
    "price",
    "current_price",
    "regularMarketPrice",
)

change = get_value(
    quote,
    "change",
    "price_change",
    "regularMarketChange",
)

change_pct = get_value(
    quote,
    "change_percent",
    "changePercent",
    "regularMarketChangePercent",
)

volume = get_value(
    quote,
    "volume",
    "regularMarketVolume",
)

company_name = get_value(
    fundamentals,
    "name",
    "longName",
    "shortName",
    default=ticker,
)

sector = get_value(
    fundamentals,
    "sector",
    default="—",
)

industry = get_value(
    fundamentals,
    "industry",
    default="—",
)


# ============================================================
# HERO
# ============================================================

change_number = safe_float(change)

change_class = "neutral"

if change_number is not None:

    if change_number > 0:
        change_class = "up"

    elif change_number < 0:
        change_class = "down"

change_prefix = ""

if change_number is not None and change_number > 0:
    change_prefix = "+"

change_pct_display = fmt_percent(
    change_pct
)

st.markdown(
    f"""
<div class="hero">

<div class="hero-symbol">
NASDAQ · {ticker}
</div>

<div class="hero-company">
{company_name}
</div>

<div class="hero-price">
{fmt_money(price)}
</div>

<div class="hero-change {change_class}">
{change_prefix}{fmt_money(change)}
&nbsp;&nbsp;·&nbsp;&nbsp;
{change_prefix}{change_pct_display}
</div>

<div class="hero-meta">

<div class="meta-pill">
{sector}
</div>

<div class="meta-pill">
{industry}
</div>

<div class="meta-pill">
Trend · {technicals.get("trend", "—")}
</div>

<div class="meta-pill">
Momentum · {technicals.get("momentum", "—")}
</div>

</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# BACKEND ERROR
# ============================================================

if market.get("error"):

    st.warning(
        "Market data is temporarily unavailable. "
        "The terminal remains online and will retry automatically."
    )


# ============================================================
# KPI ROW
# ============================================================

k1, k2, k3, k4, k5, k6 = st.columns(6)


with k1:

    st.markdown(
        f"""
        <div class="glass kpi">
        <div class="kpi-label">Market Cap</div>
        <div class="kpi-value">
        {fmt_number(get_value(fundamentals, "market_cap"))}
        </div>
        <div class="kpi-sub">Equity value</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with k2:

    pe = get_value(
        fundamentals,
        "pe",
        "trailing_pe",
    )

    st.markdown(
        f"""
        <div class="glass kpi">
        <div class="kpi-label">P / E</div>
        <div class="kpi-value">
        {fmt_number(pe)}
        </div>
        <div class="kpi-sub">Trailing valuation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with k3:

    st.markdown(
        f"""
        <div class="glass kpi">
        <div class="kpi-label">Revenue Growth</div>
        <div class="kpi-value">
        {fmt_percent(get_value(fundamentals, "revenue_growth", "revenueGrowth"))}
        </div>
        <div class="kpi-sub">YoY growth</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with k4:

    st.markdown(
        f"""
        <div class="glass kpi">
        <div class="kpi-label">ROE</div>
        <div class="kpi-value">
        {fmt_percent(get_value(fundamentals, "roe", "return_on_equity"))}
        </div>
        <div class="kpi-sub">Capital efficiency</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with k5:

    rsi_display = (
        f"{technicals['rsi']:.1f}"
        if technicals.get("rsi") is not None
        else "—"
    )

    st.markdown(
        f"""
        <div class="glass kpi">
        <div class="kpi-label">RSI</div>
        <div class="kpi-value">
        {rsi_display}
        </div>
        <div class="kpi-sub">14D momentum</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with k6:

    st.markdown(
        f"""
        <div class="glass kpi">
        <div class="kpi-label">AI Conviction</div>
        <div class="kpi-value">
        {brain["conviction"]}/100
        </div>
        <div class="kpi-sub">Research composite</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# NAVIGATION
# ============================================================

tabs = st.tabs(
    [
        "◉ Overview",
        "◈ AI Brain",
        "▣ Fundamentals",
        "⌁ Technicals",
        "⚠ Risk",
        "◎ Research",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    left, right = st.columns(
        [1.75, 1]
    )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    with left:

        st.markdown(
            '<div class="section-title">Price Intelligence</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-sub">Market structure · trend · momentum</div>',
            unsafe_allow_html=True,
        )

        if not technical_df.empty:

            chart_columns = [
                c
                for c in [
                    "Close",
                    "SMA20",
                    "SMA50",
                    "SMA200",
                ]
                if c in technical_df.columns
            ]

            st.line_chart(
                technical_df[chart_columns],
                height=420,
            )

        else:

            st.markdown(
                """
                <div class="glass"
                style="
                    height:420px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    color:rgba(255,255,255,.35);
                ">
                Waiting for market data...
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # SIGNALS
    # --------------------------------------------------------

    with right:

        st.markdown(
            '<div class="section-title">Market Signals</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="signal">
            <div class="signal-label">Trend Regime</div>
            <div class="signal-value">
            {technicals.get("trend", "—")}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            f"""
            <div class="signal">
            <div class="signal-label">Momentum</div>
            <div class="signal-value">
            {technicals.get("momentum", "—")}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            f"""
            <div class="signal">
            <div class="signal-label">Volume</div>
            <div class="signal-value">
            {fmt_number(volume)}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            f"""
            <div class="signal">
            <div class="signal-label">AI Conviction</div>
            <div class="signal-value">
            {brain["conviction"]} / 100
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
        f"""
        <div class="brain-card">

        <div class="brain-label">
        SIMON AI INVESTMENT BRAIN
        </div>

        <div class="brain-title">
        {ticker} Intelligence Matrix
        </div>

        <div class="brain-description">
        Multi-dimensional investment reasoning across
        business quality, competitive moat, growth,
        valuation, momentum and risk.
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    score_columns = st.columns(6)

    score_items = [
        ("QUALITY", brain["quality"]),
        ("MOAT", brain["moat"]),
        ("GROWTH", brain["growth"]),
        ("VALUATION", brain["valuation"]),
        ("MOMENTUM", brain["momentum"]),
        ("RISK", brain["risk"]),
    ]

    for column, (label, score) in zip(
        score_columns,
        score_items,
    ):

        with column:

            st.markdown(
                f"""
                <div class="glass score-box">

                <div class="score-number">
                {score}
                </div>

                <div class="score-caption">
                {label}
                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


    st.markdown(
        '<div class="section-title">Investment Brain</div>',
        unsafe_allow_html=True,
    )

    brain_left, brain_right = st.columns(
        [1.4, 1]
    )


    with brain_left:

        st.markdown(
            f"""
            <div class="glass">

            <div class="brain-label">
            CORE THESIS
            </div>

            <div style="
                font-size:21px;
                font-weight:850;
                margin-top:5px;
            ">
            {ticker} · Structural Intelligence
            </div>

            <div class="research"
            style="margin-top:16px;">

            <b>Business Quality</b><br>
            The current model evaluates profitability,
            capital efficiency and operating characteristics.

            <br><br>

            <b>Competitive Moat</b><br>
            Moat strength is estimated from quality,
            growth and valuation characteristics.

            <br><br>

            <b>Growth Engine</b><br>
            Revenue and earnings acceleration are incorporated
            into the growth score.

            <br><br>

            <b>Valuation</b><br>
            Current valuation is interpreted relative to
            simplified historical-style valuation bands.

            <br><br>

            <b>Risk</b><br>
            Leverage, trend deterioration and extreme momentum
            conditions increase the risk score.

            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with brain_right:

        st.markdown(
            """
            <div class="glass">

            <div class="brain-label">
            DECISION MAP
            </div>

            """,
            unsafe_allow_html=True,
        )

        conviction = brain["conviction"]

        if conviction >= 80:

            decision = "HIGH CONVICTION"
            decision_note = "Strong multi-factor alignment."

        elif conviction >= 65:

            decision = "POSITIVE BIAS"
            decision_note = "Favorable setup with caveats."

        elif conviction >= 50:

            decision = "NEUTRAL"
            decision_note = "Evidence remains mixed."

        elif conviction >= 35:

            decision = "CAUTIOUS"
            decision_note = "Risk/reward requires discipline."

        else:

            decision = "LOW CONVICTION"
            decision_note = "Weak multi-factor alignment."

        st.markdown(
            f"""
            <div style="
                font-size:29px;
                font-weight:900;
                margin-top:12px;
                letter-spacing:-1px;
            ">
            {conviction}
            </div>

            <div style="
                font-size:11px;
                color:rgba(255,255,255,.45);
                margin-top:-3px;
            ">
            / 100 CONVICTION
            </div>

            <div style="
                margin-top:18px;
                font-size:16px;
                font-weight:800;
            ">
            {decision}
            </div>

            <div style="
                margin-top:6px;
                font-size:11px;
                color:rgba(255,255,255,.45);
            ">
            {decision_note}
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
        '<div class="section-title">Fundamental Intelligence</div>',
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        st.markdown(
            f"""
            <div class="glass">

            <div class="brain-label">
            VALUATION
            </div>

            <div style="margin-top:15px;line-height:2;">

            <b>P/E</b> ·
            {fmt_number(get_value(fundamentals, "pe", "trailing_pe"))}

            <br>

            <b>Forward P/E</b> ·
            {fmt_number(get_value(fundamentals, "forward_pe"))}

            <br>

            <b>P/B</b> ·
            {fmt_number(get_value(fundamentals, "price_to_book"))}

            <br>

            <b>EV / EBITDA</b> ·
            {fmt_number(get_value(fundamentals, "ev_to_ebitda"))}

            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with f2:

        st.markdown(
            f"""
            <div class="glass">

            <div class="brain-label">
            GROWTH & PROFITABILITY
            </div>

            <div style="margin-top:15px;line-height:2;">

            <b>Revenue Growth</b> ·
            {fmt_percent(get_value(fundamentals, "revenue_growth", "revenueGrowth"))}

            <br>

            <b>Earnings Growth</b> ·
            {fmt_percent(get_value(fundamentals, "earnings_growth", "earningsGrowth"))}

            <br>

            <b>Gross Margin</b> ·
            {fmt_percent(get_value(fundamentals, "gross_margin", "grossMargin"))}

            <br>

            <b>Operating Margin</b> ·
            {fmt_percent(get_value(fundamentals, "operating_margin", "operatingMargin"))}

            <br>

            <b>ROE</b> ·
            {fmt_percent(get_value(fundamentals, "roe", "return_on_equity"))}

            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with f3:

        st.markdown(
            f"""
            <div class="glass">

            <div class="brain-label">
            BALANCE SHEET
            </div>

            <div style="margin-top:15px;line-height:2;">

            <b>Cash</b> ·
            {fmt_number(get_value(fundamentals, "total_cash"))}

            <br>

            <b>Debt</b> ·
            {fmt_number(get_value(fundamentals, "total_debt"))}

            <br>

            <b>Current Ratio</b> ·
            {fmt_number(get_value(fundamentals, "current_ratio"))}

            <br>

            <b>Debt / Equity</b> ·
            {fmt_number(get_value(fundamentals, "debt_to_equity"))}

            <br>

            <b>Free Cash Flow</b> ·
            {fmt_number(get_value(fundamentals, "free_cash_flow"))}

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
        '<div class="section-title">Technical Intelligence</div>',
        unsafe_allow_html=True,
    )

    if not technical_df.empty:

        t1, t2 = st.columns(
            [2, 1]
        )

        with t1:

            columns = [
                c
                for c in [
                    "Close",
                    "SMA20",
                    "SMA50",
                    "SMA200",
                ]
                if c in technical_df.columns
            ]

            st.line_chart(
                technical_df[columns],
                height=450,
            )

        with t2:

            st.markdown(
                f"""
                <div class="glass">

                <div class="brain-label">
                TECHNICAL SNAPSHOT
                </div>

                <div style="
                    line-height:2.3;
                    margin-top:13px;
                ">

                <b>Trend</b><br>
                {technicals.get("trend")}

                <br>

                <b>Momentum</b><br>
                {technicals.get("momentum")}

                <br>

                <b>SMA 20</b><br>
                {fmt_money(technicals.get("sma20"))}

                <br>

                <b>SMA 50</b><br>
                {fmt_money(technicals.get("sma50"))}

                <br>

                <b>SMA 200</b><br>
                {fmt_money(technicals.get("sma200"))}

                <br>

                <b>RSI</b><br>
                {rsi_display}

                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    else:

        st.info(
            "Technical data is currently unavailable."
        )


# ============================================================
# RISK
# ============================================================

with tabs[4]:

    st.markdown(
        '<div class="section-title">Risk Intelligence</div>',
        unsafe_allow_html=True,
    )

    risk = brain["risk"]

    r1, r2, r3 = st.columns(3)

    with r1:

        st.markdown(
            f"""
            <div class="glass score-box">

            <div class="score-number">
            {risk}
            </div>

            <div class="score-caption">
            RISK SCORE
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with r2:

        trend_risk = (
            "Elevated"
            if technicals.get("trend") == "BEARISH"
            else "Normal"
        )

        st.markdown(
            f"""
            <div class="glass score-box">

            <div style="
                font-size:25px;
                font-weight:850;
            ">
            {trend_risk}
            </div>

            <div class="score-caption">
            TREND RISK
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with r3:

        debt = safe_float(
            get_value(
                fundamentals,
                "debt_to_equity",
            )
        )

        balance = (
            "Watch"
            if debt is not None and debt > 150
            else "Normal"
        )

        st.markdown(
            f"""
            <div class="glass score-box">

            <div style="
                font-size:25px;
                font-weight:850;
            ">
            {balance}
            </div>

            <div class="score-caption">
            BALANCE SHEET
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    st.markdown(
        """
        <div class="glass">

        <div class="brain-label">
        RISK FRAMEWORK
        </div>

        <div style="
            line-height:1.8;
            color:rgba(255,255,255,.62);
            font-size:13px;
            margin-top:12px;
        ">

        <b>1 · Valuation Risk</b><br>
        High valuation multiples can create downside
        asymmetry when expectations reset.

        <br><br>

        <b>2 · Balance Sheet Risk</b><br>
        Leverage and liquidity conditions influence
        financial resilience.

        <br><br>

        <b>3 · Momentum Risk</b><br>
        Extremely overbought conditions can increase
        short-term volatility.

        <br><br>

        <b>4 · Trend Risk</b><br>
        Persistent price weakness can indicate deteriorating
        market structure.

        <br><br>

        <b>5 · Model Risk</b><br>
        AI scores are decision-support signals, not
        guarantees of future returns.

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
        <div class="brain-card">

        <div class="brain-label">
        PERPLEXITY-STYLE RESEARCH LAYER
        </div>

        <div class="brain-title">
        Ask the Investment Brain
        </div>

        <div class="brain-description">
        Run a multi-agent research committee combining
        business quality, valuation, growth, technical structure
        and event-driven reasoning.
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    if not AI_BACKEND:

        st.warning(
            """
            AI backend is not installed.

            Connect `ai.orchestrator` to activate the
            full Investment Committee.
            """
        )

    else:

        if ai_status.get("configured"):

            launch = st.button(
                "🧠  Launch Deep Investment Research",
                use_container_width=True,
                type="primary",
            )

            if launch:

                context = {
                    "ticker": ticker,
                    "quote": quote,
                    "fundamentals": fundamentals,
                    "technicals": technicals,
                    "brain_scores": brain,
                    "timestamp": datetime.now().isoformat(),
                }

                try:

                    st.session_state.research_running = True

                    with st.spinner(
                        "Investment Brain is assembling the committee..."
                    ):

                        report = run_full_ai_research(
                            ticker,
                            context,
                        )

                    st.session_state.ai_report = report
                    st.session_state.research_running = False

                except Exception as exc:

                    st.session_state.research_running = False

                    st.error(
                        f"Research engine error: {exc}"
                    )


            report = st.session_state.ai_report

            if report:

                st.divider()

                committee = safe_dict(
                    report.get(
                        "committee",
                        {}
                    )
                )

                st.markdown(
                    '<div class="section-title">Investment Committee</div>',
                    unsafe_allow_html=True,
                )

                if committee.get("success"):

                    st.markdown(
                        f"""
                        <div class="glass">

                        <div class="research">
                        {committee.get("content", "No committee conclusion.")}
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                else:

                    st.warning(
                        committee.get(
                            "error",
                            "Committee unavailable.",
                        )
                    )


                agents = report.get(
                    "agents",
                    [],
                )

                if agents:

                    st.markdown(
                        '<div class="section-title">Agent Matrix</div>',
                        unsafe_allow_html=True,
                    )

                    agent_cols = st.columns(
                        min(
                            len(agents),
                            4,
                        )
                    )

                    for index, agent in enumerate(
                        agents
                    ):

                        agent = safe_dict(agent)

                        with agent_cols[
                            index % len(agent_cols)
                        ]:

                            name = str(
                                agent.get(
                                    "agent",
                                    "Agent",
                                )
                            ).replace(
                                "_",
                                " ",
                            ).upper()

                            conclusion = agent.get(
                                "conclusion",
                                "No conclusion.",
                            )

                            st.markdown(
                                f"""
                                <div class="agent">

                                <div class="agent-name">
                                {name}
                                </div>

                                <div class="agent-note"
                                style="
                                    margin-top:12px;
                                    line-height:1.6;
                                ">
                                {conclusion[:650]}
                                </div>

                                </div>
                                """,
                                unsafe_allow_html=True,
                            )


                debate = safe_dict(
                    report.get(
                        "debate",
                        {},
                    )
                )

                if debate:

                    st.markdown(
                        '<div class="section-title">Bull vs Bear</div>',
                        unsafe_allow_html=True,
                    )

                    if debate.get("success"):

                        st.markdown(
                            f"""
                            <div class="glass">

                            <div class="research">
                            {debate.get("content", "")}
                            </div>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    else:

                        st.warning(
                            debate.get(
                                "error",
                                "Debate unavailable.",
                            )
                        )

        else:

            st.markdown(
                """
                <div class="glass"
                style="text-align:center;padding:45px;">

                <div style="
                    font-size:38px;
                    margin-bottom:10px;
                ">
                ◈
                </div>

                <div style="
                    font-size:20px;
                    font-weight:850;
                ">
                Investment Brain is sleeping
                </div>

                <div style="
                    color:rgba(255,255,255,.43);
                    font-size:12px;
                    margin-top:7px;
                ">
                Configure your AI provider API key
                to activate the research committee.
                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# BOTTOM INTELLIGENCE BAR
# ============================================================

st.write("")

st.markdown(
    f"""
    <div class="glass"
    style="
        padding:15px 20px;
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap:20px;
    ">

    <div>

    <span style="
        font-size:10px;
        color:rgba(255,255,255,.36);
        letter-spacing:1px;
    ">
    SIMON BRAIN
    </span>

    <span style="
        font-size:12px;
        margin-left:12px;
        font-weight:700;
    ">
    {ticker}
    </span>

    </div>

    <div style="
        font-size:10px;
        color:rgba(255,255,255,.38);
    ">
    Conviction {brain["conviction"]}/100
    &nbsp; · &nbsp;
    Trend {technicals.get("trend", "—")}
    &nbsp; · &nbsp;
    Risk {brain["risk"]}/100
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

    ◈ SIMON INVESTMENT BRAIN V14.0
    &nbsp; · &nbsp;
    AI-NATIVE US EQUITY INTELLIGENCE TERMINAL
    &nbsp; · &nbsp;
    {datetime.now().strftime("%Y-%m-%d %H:%M")}

    <br><br>

    Research and decision-support tool only.
    Not financial advice.

    </div>
    """,
    unsafe_allow_html=True,
)