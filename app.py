from __future__ import annotations

import os
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None
    make_subplots = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# SIMON STOCK — AI INVESTMENT TERMINAL V20
# Bloomberg × Perplexity × Liquid Glass
# ============================================================

st.set_page_config(
    page_title="Simon Stock — AI Investment Terminal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LIQUID GLASS DESIGN SYSTEM
# ============================================================

st.markdown(
    """
<style>

:root {
    --bg: #05070b;
    --panel: rgba(255,255,255,.055);
    --panel2: rgba(255,255,255,.035);
    --line: rgba(255,255,255,.095);
    --soft: rgba(255,255,255,.055);
    --text: #f5f7ff;
    --muted: rgba(235,241,255,.55);
    --blue: #8ca9ff;
    --cyan: #72e7ff;
    --green: #66e0a3;
    --red: #ff788d;
    --amber: #f4c86b;
    --purple: #bd9cff;
}

html, body, [class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "SF Pro Display",
        "SF Pro Text",
        "Segoe UI",
        sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 5% -10%,
            rgba(99,126,255,.22),
            transparent 28%
        ),
        radial-gradient(
            circle at 96% 4%,
            rgba(70,218,255,.11),
            transparent 24%
        ),
        radial-gradient(
            circle at 55% 100%,
            rgba(116,92,255,.07),
            transparent 28%
        ),
        linear-gradient(
            145deg,
            #04060a 0%,
            #090c13 48%,
            #05070b 100%
        );

    color: var(--text);
}

.block-container {
    max-width: 1580px;
    padding-top: 1rem;
    padding-bottom: 5rem;
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            rgba(5,8,14,.96),
            rgba(8,10,16,.92)
        );

    border-right:
        1px solid rgba(255,255,255,.07);
}

section[data-testid="stSidebar"] * {
    color: #edf2ff;
}

.stButton > button {
    border-radius: 14px !important;
    border:
        1px solid rgba(255,255,255,.10) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    padding: 5px;

    border-radius: 17px;

    background:
        rgba(255,255,255,.025);

    border:
        1px solid rgba(255,255,255,.05);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 8px 14px;
}

div[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.065),
            rgba(255,255,255,.025)
        );

    border:
        1px solid var(--line);

    border-radius: 18px;

    padding:
        13px 15px;
}


/* ============================================================
   COMMAND BAR
   ============================================================ */

.command-bar {
    display:flex;
    align-items:center;
    justify-content:space-between;

    gap:20px;

    padding:
        10px 0 17px;
}

.brand {
    font-size:28px;
    font-weight:900;
    letter-spacing:-1.5px;
}

.brand-accent {
    color:var(--blue);
}

.command-right {
    display:flex;
    align-items:center;
    gap:9px;
}

.pill {
    display:inline-flex;
    align-items:center;
    gap:7px;

    padding:
        7px 10px;

    border-radius:999px;

    border:
        1px solid rgba(255,255,255,.09);

    background:
        rgba(255,255,255,.045);

    color:
        var(--muted);

    font-size:10px;
    font-weight:800;
    letter-spacing:.7px;
}

.live {
    color:#a4f1c6;

    border-color:
        rgba(102,224,163,.18);

    background:
        rgba(102,224,163,.07);
}

.dot {
    width:6px;
    height:6px;

    border-radius:50%;

    background:
        var(--green);

    box-shadow:
        0 0 12px
        rgba(102,224,163,.8);
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    position:relative;

    overflow:hidden;

    padding:30px;

    min-height:280px;

    border-radius:32px;

    border:
        1px solid
        rgba(255,255,255,.12);

    background:
        radial-gradient(
            circle at 88% 18%,
            rgba(124,161,255,.22),
            transparent 25%
        ),
        radial-gradient(
            circle at 68% 100%,
            rgba(91,226,255,.08),
            transparent 26%
        ),
        linear-gradient(
            140deg,
            rgba(255,255,255,.095),
            rgba(255,255,255,.025)
        );

    box-shadow:
        0 30px 100px
        rgba(0,0,0,.30);

    backdrop-filter:
        blur(34px)
        saturate(140%);

    -webkit-backdrop-filter:
        blur(34px)
        saturate(140%);
}

.hero-symbol {
    color:var(--muted);

    font-size:11px;
    font-weight:850;

    letter-spacing:1.7px;
}

.hero-ticker {
    font-size:55px;

    line-height:.98;

    font-weight:950;

    letter-spacing:-3px;

    margin-top:10px;
}

.hero-company {
    color:var(--muted);

    font-size:14px;

    margin-top:8px;
}

.hero-price {
    font-size:43px;

    font-weight:950;

    letter-spacing:-2px;

    margin-top:25px;
}

.hero-change {
    font-size:14px;
    font-weight:750;
}

.up {
    color:var(--green);
}

.down {
    color:var(--red);
}

.decision {
    position:absolute;

    top:28px;
    right:30px;

    text-align:right;

    z-index:3;
}

.decision-label {
    color:var(--muted);

    font-size:10px;

    letter-spacing:1.5px;

    font-weight:850;
}

.decision-value {
    font-size:36px;

    line-height:1.05;

    font-weight:950;

    margin-top:5px;
}

.decision-score {
    color:var(--muted);

    font-size:12px;

    margin-top:8px;
}


/* ============================================================
   KPI
   ============================================================ */

.kpi-grid {
    display:grid;

    grid-template-columns:
        repeat(7, 1fr);

    gap:10px;

    margin-top:14px;
}

.kpi {
    min-height:95px;

    padding:14px 15px;

    border-radius:18px;

    border:
        1px solid
        var(--soft);

    background:
        rgba(255,255,255,.035);
}

.kpi-label {
    color:var(--muted);

    font-size:10px;

    font-weight:800;

    text-transform:uppercase;

    letter-spacing:.8px;
}

.kpi-value {
    font-size:23px;

    font-weight:900;

    letter-spacing:-.8px;

    margin-top:8px;
}

.kpi-note {
    color:var(--muted);

    font-size:10px;

    margin-top:3px;
}


/* ============================================================
   SECTION
   ============================================================ */

.section {
    margin:
        29px 0 12px 2px;

    font-size:20px;

    font-weight:900;

    letter-spacing:-.6px;
}

.section-sub {
    margin-top:-7px;

    margin-bottom:15px;

    color:var(--muted);

    font-size:12px;
}


/* ============================================================
   BRAIN
   ============================================================ */

.brain-grid {
    display:grid;

    grid-template-columns:
        repeat(5, 1fr);

    gap:11px;
}

.brain-card {
    position:relative;

    overflow:hidden;

    padding:17px;

    min-height:160px;

    border:
        1px solid
        var(--line);

    border-radius:20px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.065),
            rgba(255,255,255,.025)
        );
}

.brain-card:before {
    content:"";

    position:absolute;

    top:0;
    left:0;
    right:0;

    height:1px;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(135,170,255,.65),
            transparent
        );
}

.brain-name {
    font-size:11px;

    font-weight:850;

    letter-spacing:.8px;
}

.brain-role {
    color:var(--muted);

    font-size:10px;

    margin-top:3px;
}

.brain-score {
    font-size:34px;

    font-weight:950;

    margin-top:17px;
}

.bar {
    height:5px;

    border-radius:99px;

    background:
        rgba(255,255,255,.075);

    overflow:hidden;

    margin-top:9px;
}

.fill {
    height:100%;

    border-radius:99px;

    background:
        linear-gradient(
            90deg,
            var(--blue),
            var(--cyan)
        );
}


/* ============================================================
   CARDS
   ============================================================ */

.glass-card {
    padding:21px;

    margin-bottom:15px;

    border:
        1px solid
        var(--line);

    border-radius:22px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.06),
            rgba(255,255,255,.022)
        );

    box-shadow:
        0 18px 60px
        rgba(0,0,0,.15);

    backdrop-filter:
        blur(25px);
}

.card-title {
    font-size:16px;

    font-weight:900;

    letter-spacing:-.25px;
}

.card-meta {
    color:var(--muted);

    font-size:11px;

    margin-top:3px;
}

.thesis {
    margin-top:14px;

    padding:
        15px 17px;

    border-left:
        2px solid
        var(--blue);

    border-radius:
        0 16px 16px 0;

    background:
        rgba(125,162,255,.055);

    color:#eaf0ff;

    font-size:13px;

    line-height:1.72;
}

.fact-row {
    display:flex;

    justify-content:space-between;

    gap:20px;

    padding:10px 0;

    border-bottom:
        1px solid
        var(--soft);

    font-size:12px;
}

.fact-row:last-child {
    border-bottom:none;
}

.fact-label {
    color:var(--muted);
}


/* ============================================================
   SIGNALS
   ============================================================ */

.signal {
    display:inline-flex;

    align-items:center;

    padding:
        5px 8px;

    border-radius:999px;

    font-size:9px;

    font-weight:900;

    letter-spacing:.65px;

    margin:
        3px 4px 3px 0;
}

.good {
    color:#9af0c1;

    background:
        rgba(102,224,163,.09);
}

.warn {
    color:#f7d88f;

    background:
        rgba(244,200,107,.09);
}

.bad {
    color:#ff9cab;

    background:
        rgba(255,120,141,.09);
}

.info {
    color:#9ab8ff;

    background:
        rgba(140,169,255,.09);
}


/* ============================================================
   AI COPILOT
   ============================================================ */

.ai-shell {
    padding:22px;

    border:
        1px solid
        rgba(140,169,255,.18);

    border-radius:24px;

    background:
        radial-gradient(
            circle at 85% 5%,
            rgba(114,231,255,.10),
            transparent 25%
        ),
        linear-gradient(
            145deg,
            rgba(117,146,255,.10),
            rgba(255,255,255,.025)
        );
}

.ai-head {
    display:flex;

    justify-content:space-between;

    gap:15px;

    align-items:flex-start;
}

.ai-title {
    font-size:20px;

    font-weight:950;
}

.ai-sub {
    color:var(--muted);

    font-size:11px;

    margin-top:3px;
}

.ai-answer {
    margin-top:16px;

    padding:17px;

    border:
        1px solid
        rgba(255,255,255,.07);

    border-radius:18px;

    background:
        rgba(0,0,0,.12);

    line-height:1.7;

    font-size:13px;
}


/* ============================================================
   SCENARIO
   ============================================================ */

.scenario-grid {
    display:grid;

    grid-template-columns:
        repeat(3,1fr);

    gap:12px;
}

.scenario {
    border:
        1px solid
        var(--line);

    border-radius:18px;

    padding:17px;

    background:
        rgba(255,255,255,.035);

    min-height:165px;
}

.scenario-name {
    font-size:12px;

    font-weight:900;
}

.scenario-prob {
    color:var(--muted);

    font-size:10px;

    margin-top:2px;
}

.scenario-value {
    font-size:27px;

    font-weight:950;

    margin-top:16px;
}


/* ============================================================
   RISK
   ============================================================ */

.risk-meter {
    height:9px;

    border-radius:99px;

    background:
        rgba(255,255,255,.07);

    overflow:hidden;

    margin-top:10px;
}

.risk-fill {
    height:100%;

    border-radius:99px;

    background:
        linear-gradient(
            90deg,
            var(--green),
            var(--amber),
            var(--red)
        );
}


/* ============================================================
   SOURCE CHIPS
   ============================================================ */

.source-chip {
    display:inline-block;

    margin:
        3px 4px 3px 0;

    padding:
        5px 8px;

    border-radius:9px;

    background:
        rgba(255,255,255,.045);

    color:var(--muted);

    font-size:9px;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media(max-width:1100px) {

    .kpi-grid {
        grid-template-columns:
            repeat(4,1fr);
    }

    .brain-grid {
        grid-template-columns:
            repeat(3,1fr);
    }
}

@media(max-width:760px) {

    .kpi-grid {
        grid-template-columns:
            repeat(2,1fr);
    }

    .brain-grid {
        grid-template-columns:
            repeat(2,1fr);
    }

    .scenario-grid {
        grid-template-columns:
            1fr;
    }

    .decision {
        position:static;

        text-align:left;

        margin-top:25px;
    }

    .hero-ticker {
        font-size:43px;
    }

    .hero {
        min-height:auto;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# UTILS
# ============================================================

def num(v):
    try:
        if v is None:
            return None

        x = float(v)

        if math.isnan(x) or math.isinf(x):
            return None

        return x

    except Exception:
        return None


def money(v, digits=2):
    x = num(v)

    if x is None:
        return "—"

    return f"${x:,.{digits}f}"


def pct(v, digits=1):
    x = num(v)

    if x is None:
        return "—"

    return f"{x * 100:.{digits}f}%"


def multiple(v):
    x = num(v)

    if x is None:
        return "—"

    return f"{x:.1f}×"


def compact(v):
    x = num(v)

    if x is None:
        return "—"

    a = abs(x)

    if a >= 1e12:
        return f"{x/1e12:.2f}T"

    if a >= 1e9:
        return f"{x/1e9:.2f}B"

    if a >= 1e6:
        return f"{x/1e6:.2f}M"

    if a >= 1e3:
        return f"{x/1e3:.1f}K"

    return f"{x:.0f}"


def clamp(x, lo=0, hi=100):
    return int(np.clip(x, lo, hi))


def safe_get(d, *keys):
    for key in keys:

        if key in d and d[key] is not None:
            return d[key]

    return None


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_stock(ticker, period):

    if yf is None:
        raise RuntimeError(
            "yfinance 未安装。"
        )

    stock = yf.Ticker(ticker)

    history = stock.history(
        period=period,
        interval="1d",
        auto_adjust=False,
    )

    if history is None or history.empty:
        raise RuntimeError(
            f"无法找到 {ticker} 的行情数据。"
        )

    try:
        info = stock.info or {}

    except Exception:
        info = {}

    return history, info


# ============================================================
# TECHNICAL ENGINE
# ============================================================

def technical_engine(df):

    data = df.copy()

    close = data["Close"].astype(float)

    data["SMA20"] = close.rolling(20).mean()
    data["SMA50"] = close.rolling(50).mean()
    data["SMA200"] = close.rolling(200).mean()

    data["EMA12"] = close.ewm(
        span=12,
        adjust=False
    ).mean()

    data["EMA26"] = close.ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = (
        data["EMA12"] -
        data["EMA26"]
    )

    data["MACD_SIGNAL"] = (
        data["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    delta = close.diff()

    gain = delta.clip(
        lower=0
    ).rolling(14).mean()

    loss = (
        -delta.clip(
            upper=0
        )
        .rolling(14)
        .mean()
    )

    rs = gain / loss.replace(
        0,
        np.nan
    )

    data["RSI"] = (
        100 -
        100 / (1 + rs)
    )

    data["TR"] = pd.concat(
        [
            data["High"] - data["Low"],
            (
                data["High"] -
                data["Close"].shift()
            ).abs(),
            (
                data["Low"] -
                data["Close"].shift()
            ).abs(),
        ],
        axis=1
    ).max(axis=1)

    data["ATR"] = (
        data["TR"]
        .rolling(14)
        .mean()
    )

    return data


def get_technical_state(df):

    if df.empty:
        return {}

    last = df.iloc[-1]

    price = num(last["Close"])
    sma20 = num(last["SMA20"])
    sma50 = num(last["SMA50"])
    sma200 = num(last["SMA200"])
    rsi = num(last["RSI"])
    macd = num(last["MACD"])
    macd_signal = num(
        last["MACD_SIGNAL"]
    )
    atr = num(last["ATR"])

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

        elif macd < macd_signal:
            macd_state = "NEGATIVE"

    volatility = None

    if (
        atr is not None
        and price is not None
        and price != 0
    ):
        volatility = atr / price

    return {
        "trend": trend,
        "momentum": momentum,
        "macd": macd_state,
        "rsi": rsi,
        "atr": atr,
        "volatility": volatility,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
    }


# ============================================================
# INVESTMENT BRAIN
# ============================================================

def calculate_brain(info, technicals):

    revenue_growth = num(
        safe_get(
            info,
            "revenueGrowth"
        )
    )

    earnings_growth = num(
        safe_get(
            info,
            "earningsGrowth"
        )
    )

    gross_margin = num(
        safe_get(
            info,
            "grossMargins"
        )
    )

    operating_margin = num(
        safe_get(
            info,
            "operatingMargins"
        )
    )

    roe = num(
        safe_get(
            info,
            "returnOnEquity"
        )
    )

    debt = num(
        safe_get(
            info,
            "debtToEquity"
        )
    )

    pe = num(
        safe_get(
            info,
            "trailingPE"
        )
    )

    forward_pe = num(
        safe_get(
            info,
            "forwardPE"
        )
    )

    # --------------------------
    # VALUE
    # --------------------------

    value = 55

    if pe is not None:

        if pe < 15:
            value += 20

        elif pe < 25:
            value += 10

        elif pe > 45:
            value -= 18

        elif pe > 30:
            value -= 8

    if forward_pe is not None:

        if (
            pe is not None
            and forward_pe < pe
        ):
            value += 8

        elif (
            pe is not None
            and forward_pe > pe * 1.1
        ):
            value -= 8

    value = clamp(value)


    # --------------------------
    # BUSINESS
    # --------------------------

    business = 50

    if gross_margin is not None:

        business += clamp(
            gross_margin * 50,
            -20,
            25
        )

    if operating_margin is not None:

        business += clamp(
            operating_margin * 60,
            -15,
            25
        )

    if roe is not None:

        business += clamp(
            roe * 30,
            -10,
            20
        )

    business = clamp(
        business
    )


    # --------------------------
    # GROWTH
    # --------------------------

    growth = 50

    if revenue_growth is not None:

        growth += clamp(
            revenue_growth * 80,
            -20,
            25
        )

    if earnings_growth is not None:

        growth += clamp(
            earnings_growth * 70,
            -20,
            25
        )

    growth = clamp(
        growth
    )


    # --------------------------
    # EVENT / MOMENTUM
    # --------------------------

    event = 50

    if technicals.get(
        "trend"
    ) == "BULLISH":

        event += 20

    elif technicals.get(
        "trend"
    ) == "BEARISH":

        event -= 20

    if technicals.get(
        "momentum"
    ) == "POSITIVE":

        event += 12

    elif technicals.get(
        "momentum"
    ) == "NEGATIVE":

        event -= 12

    event = clamp(
        event
    )


    # --------------------------
    # RISK
    # --------------------------

    risk = 50

    if debt is not None:

        if debt > 150:
            risk += 22

        elif debt > 100:
            risk += 12

        elif debt < 50:
            risk -= 12

    if technicals.get(
        "trend"
    ) == "BEARISH":

        risk += 12

    if technicals.get(
        "momentum"
    ) == "OVERBOUGHT":

        risk += 8

    risk = clamp(
        risk
    )


    # --------------------------
    # COMPOSITE
    # --------------------------

    conviction = (
        value * .25
        + business * .25
        + growth * .20
        + event * .15
        + (100 - risk) * .15
    )

    conviction = clamp(
        conviction
    )

    if conviction >= 72:
        decision = "BUY"

    elif conviction >= 57:
        decision = "WATCH"

    else:
        decision = "AVOID"

    return {
        "value": value,
        "business": business,
        "growth": growth,
        "event": event,
        "risk": risk,
        "conviction": conviction,
        "decision": decision,
    }


# ============================================================
# AI PROMPT ENGINE
# ============================================================

def build_ai_context(
    ticker,
    info,
    technicals,
    brain
):

    context = {

        "ticker": ticker,

        "company": safe_get(
            info,
            "longName",
            "shortName"
        ),

        "sector": info.get(
            "sector"
        ),

        "industry": info.get(
            "industry"
        ),

        "price": safe_get(
            info,
            "currentPrice",
            "regularMarketPrice"
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

        "technicals": technicals,

        "brain": brain,
    }

    return context


def local_ai_answer(
    question,
    ticker,
    info,
    technicals,
    brain
):

    company = safe_get(
        info,
        "longName",
        "shortName"
    ) or ticker

    decision = brain[
        "decision"
    ]

    conviction = brain[
        "conviction"
    ]

    trend = technicals.get(
        "trend",
        "NEUTRAL"
    )

    rsi = technicals.get(
        "rsi"
    )

    pe = info.get(
        "trailingPE"
    )

    growth = info.get(
        "revenueGrowth"
    )

    if "为什么" in question or "why" in question.lower():

        return (
            f"{company} 当前的核心判断是 "
            f"{decision}，Conviction {conviction}/100。"
            f"主要依据是商业质量、成长性、估值与趋势的综合结果。"
            f"目前技术趋势为 {trend}，"
            f"收入增长约 {pct(growth)}，"
            f"P/E 为 {multiple(pe)}。"
            f"需要注意的是，这只是研究框架，不代表未来价格一定上涨。"
        )

    if "风险" in question or "risk" in question.lower():

        return (
            f"{ticker} 当前 Risk Score 为 "
            f"{brain['risk']}/100。"
            f"主要风险来自估值、资产负债表、"
            f"市场趋势以及预期变化。"
            f"如果趋势进一步转弱，或盈利增长低于市场预期，"
            f"当前投资逻辑需要重新评估。"
        )

    if "估值" in question or "valuation" in question.lower():

        return (
            f"{ticker} 的估值 Brain Score 为 "
            f"{brain['value']}/100。"
            f"当前 P/E 为 {multiple(pe)}。"
            f"如果未来盈利增速继续维持，较高估值可能具有一定容忍度；"
            f"反之，盈利预期下降会显著压缩安全边际。"
        )

    return (
        f"Investment Brain 已完成对 {ticker} 的第一轮扫描。"
        f"综合结论：{decision}，Conviction {conviction}/100。"
        f"当前趋势 {trend}，"
        f"RSI {rsi:.1f}。"
        if rsi is not None
        else
        f"Investment Brain 已完成对 {ticker} 的第一轮扫描。"
        f"综合结论：{decision}，Conviction {conviction}/100。"
    )


def run_openai(
    question,
    context
):

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if (
        not api_key
        or OpenAI is None
    ):
        return None

    model = os.getenv(
        "OPENAI_MODEL",
        "gpt-4.1-mini"
    )

    client = OpenAI(
        api_key=api_key
    )

    system = """
You are Simon Stock Investment Brain.

You are not a stock-picking chatbot.
You are an institutional-style investment research system.

Analyze companies through five lenses:

1. VALUE
2. BUSINESS QUALITY
3. FIRST PRINCIPLES GROWTH
4. MARKET / EVENT
5. RED TEAM RISK

Rules:

- Separate facts from assumptions.
- Never invent unavailable data.
- Explain the reasoning chain.
- Identify what could make the thesis wrong.
- Avoid pretending to know future prices.
- Be concise but intellectually rigorous.
- The output is research assistance, not financial advice.

Return:
Verdict
Why
Bull Case
Bear Case
Key Risks
What Changes My Mind
"""

    user = f"""
Ticker:
{context["ticker"]}

Company:
{context["company"]}

Investment Brain:
{context["brain"]}

Technical State:
{context["technicals"]}

Fundamental Data:
{context}

Question:
{question}
"""

    try:

        response = client.responses.create(
            model=model,
            instructions=system,
            input=user,
        )

        return response.output_text

    except Exception as exc:

        return (
            "AI API 调用失败："
            + str(exc)
        )


# ============================================================
# SESSION
# ============================================================

if "ticker" not in st.session_state:
    st.session_state.ticker = "NVDA"

if "period" not in st.session_state:
    st.session_state.period = "2y"

if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []

if "watchlist" not in st.session_state:
    st.session_state.watchlist = [
        "NVDA",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "TSLA",
    ]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:24px;
            font-weight:900;
            letter-spacing:-1px;
            margin-bottom:3px;
        ">
        ◈ Simon Stock
        </div>

        <div style="
            color:rgba(235,241,255,.45);
            font-size:10px;
            letter-spacing:1px;
        ">
        AI INVESTMENT TERMINAL
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    ticker_input = st.text_input(
        "Command / Ticker",
        value=st.session_state.ticker,
        placeholder="NVDA / AAPL / MSFT",
    )

    ticker = (
        ticker_input.strip().upper()
        if ticker_input
        else "NVDA"
    )

    st.session_state.ticker = ticker

    period = st.selectbox(
        "Historical Window",
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
    )

    st.session_state.period = period

    st.divider()

    st.markdown(
        "### Watchlist"
    )

    for item in st.session_state.watchlist:

        if st.button(
            item,
            key=f"watch_{item}",
            use_container_width=True
        ):
            st.session_state.ticker = item
            st.rerun()

    st.divider()

    st.markdown(
        "### Terminal Modules"
    )

    st.caption(
        "◉ Market Intelligence"
    )

    st.caption(
        "◉ Investment Brain"
    )

    st.caption(
        "◉ Valuation Lab"
    )

    st.caption(
        "◉ Risk Engine"
    )

    st.caption(
        "◉ AI Copilot"
    )

    st.divider()

    st.caption(
        "Market data · Yahoo Finance"
    )

    if os.getenv(
        "OPENAI_API_KEY"
    ):
        st.success(
            "AI Brain Connected"
        )
    else:
        st.warning(
            "Local Brain Mode"
        )


# ============================================================
# LOAD
# ============================================================

try:

    with st.spinner(
        f"Connecting to {ticker}..."
    ):

        history, info = load_stock(
            ticker,
            period
        )

except Exception as exc:

    st.error(
        f"Terminal connection failed: {exc}"
    )

    st.stop()


technical_df = technical_engine(
    history
)

technicals = get_technical_state(
    technical_df
)

brain = calculate_brain(
    info,
    technicals
)


# ============================================================
# MARKET SNAPSHOT
# ============================================================

latest = history.iloc[-1]

price = num(
    latest["Close"]
)

previous = (
    num(history["Close"].iloc[-2])
    if len(history) > 1
    else None
)

change = (
    price - previous
    if price is not None
    and previous is not None
    else None
)

change_pct = (
    change / previous
    if change is not None
    and previous
    else None
)

company = safe_get(
    info,
    "longName",
    "shortName"
) or ticker


# ============================================================
# TOP COMMAND BAR
# ============================================================

now = datetime.now(
    timezone.utc
)

st.markdown(
    f"""
<div class="command-bar">

<div class="brand">
◈ <span class="brand-accent">Simon</span> Stock
</div>

<div class="command-right">

<div class="pill live">
<span class="dot"></span>
MARKET DATA
</div>

<div class="pill">
AI BRAIN
</div>

<div class="pill">
{now.strftime("%H:%M UTC")}
</div>

</div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

decision_color = (
    "up"
    if brain["decision"] == "BUY"
    else
    "down"
    if brain["decision"] == "AVOID"
    else
    ""
)

st.markdown(
    f"""
<div class="hero">

<div class="hero-symbol">
US EQUITY · INVESTMENT TERMINAL
</div>

<div class="hero-ticker">
{ticker}
</div>

<div class="hero-company">
{company}
</div>

<div class="hero-price">
{money(price)}
</div>

<div class="hero-change {decision_color}">
{
    ("+" if change is not None and change >= 0 else "")
    + money(change)
    if change is not None
    else "—"
}
&nbsp;&nbsp;
{
    ("+" if change_pct is not None and change_pct >= 0 else "")
    + pct(change_pct)
    if change_pct is not None
    else "—"
}
</div>

<div class="decision">

<div class="decision-label">
INVESTMENT COMMITTEE
</div>

<div class="decision-value {decision_color}">
{brain["decision"]}
</div>

<div class="decision-score">
Conviction {brain["conviction"]}/100
</div>

</div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# KPI STRIP
# ============================================================

market_cap = info.get(
    "marketCap"
)

pe = info.get(
    "trailingPE"
)

forward_pe = info.get(
    "forwardPE"
)

revenue_growth = info.get(
    "revenueGrowth"
)

roe = info.get(
    "returnOnEquity"
)

fcf = info.get(
    "freeCashflow"
)

st.markdown(
    f"""
<div class="kpi-grid">

<div class="kpi">
<div class="kpi-label">Market Cap</div>
<div class="kpi-value">{compact(market_cap)}</div>
<div class="kpi-note">Equity Value</div>
</div>

<div class="kpi">
<div class="kpi-label">P / E</div>
<div class="kpi-value">{multiple(pe)}</div>
<div class="kpi-note">Trailing</div>
</div>

<div class="kpi">
<div class="kpi-label">Forward P/E</div>
<div class="kpi-value">{multiple(forward_pe)}</div>
<div class="kpi-note">Consensus proxy</div>
</div>

<div class="kpi">
<div class="kpi-label">Revenue Growth</div>
<div class="kpi-value">{pct(revenue_growth)}</div>
<div class="kpi-note">YoY</div>
</div>

<div class="kpi">
<div class="kpi-label">ROE</div>
<div class="kpi-value">{pct(roe)}</div>
<div class="kpi-note">Profitability</div>
</div>

<div class="kpi">
<div class="kpi-label">Free Cash Flow</div>
<div class="kpi-value">{compact(fcf)}</div>
<div class="kpi-note">TTM</div>
</div>

<div class="kpi">
<div class="kpi-label">RSI</div>
<div class="kpi-value">
{
    f"{technicals['rsi']:.1f}"
    if technicals.get("rsi") is not None
    else "—"
}
</div>
<div class="kpi-note">14D Momentum</div>
</div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# INVESTMENT BRAIN
# ============================================================

st.markdown(
    '<div class="section">Investment Brain</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="section-sub">
Five independent reasoning engines → one investment committee.
</div>
""",
    unsafe_allow_html=True
)

brain_cards = [
    (
        "VALUE",
        "Margin of Safety",
        brain["value"]
    ),
    (
        "BUSINESS",
        "Economic Moat",
        brain["business"]
    ),
    (
        "GROWTH",
        "First Principles",
        brain["growth"]
    ),
    (
        "EVENT",
        "Market Catalyst",
        brain["event"]
    ),
    (
        "RED TEAM",
        "Thesis Risk",
        100 - brain["risk"]
    ),
]

cards_html = ""

for name, role, score in brain_cards:

    cards_html += f"""
    <div class="brain-card">

        <div class="brain-name">
            {name}
        </div>

        <div class="brain-role">
            {role}
        </div>

        <div class="brain-score">
            {score}
        </div>

        <div class="bar">
            <div
                class="fill"
                style="width:{score}%"
            ></div>
        </div>

    </div>
    """

st.markdown(
    f'<div class="brain-grid">{cards_html}</div>',
    unsafe_allow_html=True
)


# ============================================================
# MAIN TABS
# ============================================================

tabs = st.tabs(
    [
        "◈ Terminal",
        "🧠 Investment Brain",
        "💰 Valuation",
        "⚠ Risk Lab",
        "✦ AI Copilot",
    ]
)


# ============================================================
# TERMINAL
# ============================================================

with tabs[0]:

    left, right = st.columns(
        [1.7, 1]
    )

    with left:

        st.markdown(
            '<div class="section">Market Intelligence</div>',
            unsafe_allow_html=True
        )

        if go is not None:

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=.05,
                row_heights=[.75, .25],
            )

            fig.add_trace(
                go.Candlestick(
                    x=technical_df.index,
                    open=technical_df["Open"],
                    high=technical_df["High"],
                    low=technical_df["Low"],
                    close=technical_df["Close"],
                    name="Price",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=technical_df.index,
                    y=technical_df["SMA20"],
                    name="SMA20",
                    line=dict(
                        width=1.3
                    ),
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=technical_df.index,
                    y=technical_df["SMA50"],
                    name="SMA50",
                    line=dict(
                        width=1.3
                    ),
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=technical_df.index,
                    y=technical_df["SMA200"],
                    name="SMA200",
                    line=dict(
                        width=1.3
                    ),
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Bar(
                    x=technical_df.index,
                    y=technical_df["Volume"],
                    name="Volume",
                    opacity=.35,
                ),
                row=2,
                col=1,
            )

            fig.update_layout(
                height=540,
                margin=dict(
                    l=0,
                    r=0,
                    t=15,
                    b=0
                ),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_rangeslider_visible=False,
                legend=dict(
                    orientation="h",
                    y=1.04,
                ),
            )

            fig.update_xaxes(
                gridcolor="rgba(255,255,255,.045)"
            )

            fig.update_yaxes(
                gridcolor="rgba(255,255,255,.045)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displaylogo": False
                }
            )

        else:

            st.line_chart(
                technical_df[
                    [
                        "Close",
                        "SMA20",
                        "SMA50",
                        "SMA200",
                    ]
                ],
                height=500
            )

    with right:

        st.markdown(
            """
            <div class="glass-card">
            <div class="card-title">
            Market State
            </div>

            <div class="card-meta">
            Technical engine
            </div>
            """,
            unsafe_allow_html=True
        )

        trend_class = (
            "good"
            if technicals["trend"] == "BULLISH"
            else
            "bad"
            if technicals["trend"] == "BEARISH"
            else
            "warn"
        )

        momentum_class = (
            "good"
            if technicals["momentum"] == "POSITIVE"
            else
            "bad"
            if technicals["momentum"] == "NEGATIVE"
            else
            "warn"
        )

        st.markdown(
            f"""
            <span class="signal {trend_class}">
            {technicals["trend"]}
            </span>

            <span class="signal {momentum_class}">
            {technicals["momentum"]}
            </span>

            <div class="fact-row">
                <span class="fact-label">SMA 20</span>
                <span>{money(technicals.get("sma20"))}</span>
            </div>

            <div class="fact-row">
                <span class="fact-label">SMA 50</span>
                <span>{money(technicals.get("sma50"))}</span>
            </div>

            <div class="fact-row">
                <span class="fact-label">SMA 200</span>
                <span>{money(technicals.get("sma200"))}</span>
            </div>

            <div class="fact-row">
                <span class="fact-label">RSI</span>
                <span>
                {
                    f"{technicals['rsi']:.1f}"
                    if technicals.get("rsi") is not None
                    else "—"
                }
                </span>
            </div>

            <div class="fact-row">
                <span class="fact-label">MACD</span>
                <span>{technicals.get("macd")}</span>
            </div>

            <div class="fact-row">
                <span class="fact-label">ATR</span>
                <span>{money(technicals.get("atr"))}</span>
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# INVESTMENT BRAIN TAB
# ============================================================

with tabs[1]:

    st.markdown(
        '<div class="section">The Investment Committee</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="glass-card">

        <div class="card-title">
        {brain["decision"]} · Conviction {brain["conviction"]}/100
        </div>

        <div class="card-meta">
        Decision generated from independent investment lenses.
        </div>

        <div class="thesis">

        <b>Core Thesis</b><br><br>

        {company} 当前的 Investment Brain
        将估值、商业质量、增长、市场趋势与风险进行交叉验证。

        当前综合判断为
        <b>{brain["decision"]}</b>，
        Conviction 为
        <b>{brain["conviction"]}/100</b>。

        这不是价格预测，而是对
        “当前价格下，投资逻辑是否值得继续研究”
        的结构化判断。

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            <div class="glass-card">

            <div class="card-title">
            Bull Case
            </div>

            <div class="card-meta">
            What needs to go right
            </div>

            <div class="thesis">

            • Revenue growth remains strong.<br>
            • Operating leverage improves.<br>
            • Market continues to reward earnings visibility.<br>
            • Valuation remains supported by future cash generation.<br>
            • Competitive moat becomes stronger rather than weaker.

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="glass-card">

            <div class="card-title">
            Bear Case
            </div>

            <div class="card-meta">
            What can break the thesis
            </div>

            <div class="thesis">

            • Growth decelerates faster than expected.<br>
            • Valuation multiple compresses.<br>
            • Competitive intensity increases.<br>
            • Capital requirements rise.<br>
            • Macro conditions reduce risk appetite.

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section">Thesis Killers</div>',
        unsafe_allow_html=True
    )

    killers = [
        (
            "Valuation",
            "如果盈利增长不足以支撑当前估值，安全边际会快速下降。"
        ),
        (
            "Growth",
            "收入和利润增速持续低于市场预期时，投资逻辑需要重估。"
        ),
        (
            "Competition",
            "如果竞争对手开始侵蚀定价权或利润率，商业护城河可能减弱。"
        ),
        (
            "Macro",
            "利率、衰退、流动性和风险偏好变化可能压缩估值。"
        ),
    ]

    for title, text in killers:

        st.markdown(
            f"""
            <div class="glass-card">

            <div class="card-title">
            {title}
            </div>

            <div class="thesis">
            {text}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# VALUATION
# ============================================================

with tabs[2]:

    st.markdown(
        '<div class="section">Valuation Laboratory</div>',
        unsafe_allow_html=True
    )

    valuation_score = brain["value"]

    v1, v2, v3 = st.columns(3)

    with v1:

        st.metric(
            "Valuation Score",
            f"{valuation_score}/100"
        )

    with v2:

        st.metric(
            "Trailing P/E",
            multiple(pe)
        )

    with v3:

        st.metric(
            "Forward P/E",
            multiple(forward_pe)
        )

    st.markdown(
        """
        <div class="glass-card">

        <div class="card-title">
        Margin of Safety
        </div>

        <div class="card-meta">
        Multi-factor valuation interpretation
        </div>

        <div class="thesis">

        这里不是简单判断“PE 高还是低”。

        Investment Brain 会把当前估值与
        盈利增长、利润率、资本回报率、
        现金流能力以及未来预期放在一起。

        <br><br>

        <b>核心问题：</b>

        “这家公司未来产生的现金流，
        是否足以合理支撑今天支付的价格？”

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    valuation_data = pd.DataFrame(
        {
            "Metric": [
                "P/E",
                "Forward P/E",
                "Revenue Growth",
                "Operating Margin",
                "ROE",
            ],

            "Value": [
                multiple(pe),
                multiple(forward_pe),
                pct(
                    info.get(
                        "revenueGrowth"
                    )
                ),
                pct(
                    info.get(
                        "operatingMargins"
                    )
                ),
                pct(
                    info.get(
                        "returnOnEquity"
                    )
                ),
            ]
        }
    )

    st.dataframe(
        valuation_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RISK LAB
# ============================================================

with tabs[3]:

    st.markdown(
        '<div class="section">Risk Laboratory</div>',
        unsafe_allow_html=True
    )

    risk = brain["risk"]

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Risk Score",
            f"{risk}/100"
        )

    with r2:

        st.metric(
            "Balance Sheet",
            (
                "WATCH"
                if (
                    num(
                        info.get(
                            "debtToEquity"
                        )
                    ) or 0
                ) > 150
                else "NORMAL"
            )
        )

    with r3:

        st.metric(
            "Trend",
            technicals["trend"]
        )

    st.markdown(
        f"""
        <div class="glass-card">

        <div class="card-title">
        Systemic Risk Monitor
        </div>

        <div class="card-meta">
        Current estimated risk exposure
        </div>

        <div class="risk-meter">
            <div
                class="risk-fill"
                style="width:{risk}%"
            ></div>
        </div>

        <div class="thesis">

        Risk Score = {risk}/100

        <br><br>

        风险评分不是“股票会不会跌”的预测。

        它衡量的是当前投资逻辑中，
        有多少部分依赖高估值、强增长、
        有利趋势或低风险宏观环境。

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    risk_items = [
        (
            "Valuation Risk",
            "估值越高，对未来增长的依赖越强。"
        ),
        (
            "Growth Risk",
            "增长预期下降可能带来盈利和估值双杀。"
        ),
        (
            "Balance Sheet Risk",
            "高杠杆环境下，现金流安全边际更加重要。"
        ),
        (
            "Momentum Risk",
            "趋势转弱可能导致短期风险快速放大。"
        ),
    ]

    for title, desc in risk_items:

        st.markdown(
            f"""
            <div class="glass-card">

            <div class="card-title">
            {title}
            </div>

            <div class="thesis">
            {desc}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# AI COPILOT
# ============================================================

with tabs[4]:

    st.markdown(
        """
        <div class="ai-shell">

        <div class="ai-head">

        <div>

        <div class="ai-title">
        ✦ Investment Copilot
        </div>

        <div class="ai-sub">
        Ask the Investment Brain anything about this company.
        </div>

        </div>

        <div class="pill live">
        BRAIN ONLINE
        </div>

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    quick1, quick2, quick3, quick4 = st.columns(4)

    with quick1:

        if st.button(
            "Why this stock?",
            use_container_width=True
        ):

            st.session_state.ai_messages.append(
                "Why this stock?"
            )

    with quick2:

        if st.button(
            "What are the risks?",
            use_container_width=True
        ):

            st.session_state.ai_messages.append(
                "What are the main risks?"
            )

    with quick3:

        if st.button(
            "Is valuation attractive?",
            use_container_width=True
        ):

            st.session_state.ai_messages.append(
                "Is the valuation attractive?"
            )

    with quick4:

        if st.button(
            "What changes my mind?",
            use_container_width=True
        ):

            st.session_state.ai_messages.append(
                "What would change the investment thesis?"
            )

    question = st.chat_input(
        f"Ask Investment Brain about {ticker}..."
    )

    if question:

        st.session_state.ai_messages.append(
            question
        )

    context = build_ai_context(
        ticker,
        info,
        technicals,
        brain
    )

    for question in st.session_state.ai_messages[-6:]:

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )

        answer = run_openai(
            question,
            context
        )

        if answer is None:

            answer = local_ai_answer(
                question,
                ticker,
                info,
                technicals,
                brain
            )

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                answer
            )


# ============================================================
# SCENARIO ENGINE
# ============================================================

st.markdown(
    '<div class="section">Scenario Engine</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="section-sub">
Think in distributions, not single-point price predictions.
</div>
""",
    unsafe_allow_html=True
)

current_price = price or 0

scenario_data = [
    (
        "BULL",
        "25%",
        current_price * 1.35,
        "Growth remains above expectations."
    ),
    (
        "BASE",
        "50%",
        current_price * 1.10,
        "Growth normalizes but thesis holds."
    ),
    (
        "BEAR",
        "25%",
        current_price * .72,
        "Growth slows and multiple compresses."
    ),
]

scenario_html = ""

for name, probability, target, description in scenario_data:

    scenario_html += f"""
    <div class="scenario">

        <div class="scenario-name">
        {name}
        </div>

        <div class="scenario-prob">
        Probability {probability}
        </div>

        <div class="scenario-value">
        {money(target)}
        </div>

        <div class="card-meta">
        {description}
        </div>

    </div>
    """

st.markdown(
    f"""
    <div class="scenario-grid">
    {scenario_html}
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    f"""
    <div style="
        display:flex;
        justify-content:space-between;
        color:rgba(235,241,255,.38);
        font-size:10px;
        padding-bottom:20px;
    ">

    <span>
    ◈ SIMON STOCK V20 · AI INVESTMENT TERMINAL
    </span>

    <span>
    {ticker} · {datetime.now().strftime("%Y-%m-%d %H:%M")}
    </span>

    </div>

    <div style="
        color:rgba(235,241,255,.30);
        font-size:9px;
        text-align:center;
    ">
    Research assistance only · Not financial advice
    </div>
    """,
    unsafe_allow_html=True
)