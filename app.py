from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from data.market import (
    get_history,
    get_quote,
    get_fundamental_snapshot,
    get_market_bundle,
    data_health_check,
)
from ai.orchestrator import (
    ai_health_check,
    run_full_ai_research,
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Simon Stock V13.1",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

:root {
    --ss-radius: 24px;
}

/* ---------- App ---------- */

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(80,120,255,.10),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(120,180,255,.08),
            transparent 28%
        ),
        var(--background-color);
}

/* ---------- Header ---------- */

.ss-header {
    padding: 10px 4px 24px 4px;
}

.ss-brand {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -1.2px;
}

.ss-subtitle {
    opacity: .65;
    font-size: 14px;
    margin-top: 3px;
}

/* ---------- Glass Cards ---------- */

.ss-card {
    border: 1px solid rgba(128,128,128,.16);
    border-radius: var(--ss-radius);
    padding: 22px;
    margin-bottom: 16px;
    background: rgba(128,128,128,.07);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    box-shadow:
        0 8px 35px rgba(0,0,0,.06);
}

.ss-card-title {
    font-size: 14px;
    font-weight: 700;
    opacity: .65;
    margin-bottom: 7px;
}

.ss-card-value {
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -.8px;
}

.ss-card-small {
    font-size: 13px;
    opacity: .58;
}

/* ---------- Hero ---------- */

.ss-hero {
    border-radius: 30px;
    padding: 30px;
    margin-bottom: 20px;
    border: 1px solid rgba(128,128,128,.16);
    background:
        linear-gradient(
            135deg,
            rgba(128,160,255,.15),
            rgba(128,128,128,.05)
        );
    backdrop-filter: blur(30px);
}

.ss-hero-symbol {
    font-size: 15px;
    font-weight: 700;
    opacity: .65;
}

.ss-hero-name {
    font-size: 38px;
    font-weight: 850;
    letter-spacing: -1.5px;
}

.ss-hero-price {
    font-size: 44px;
    font-weight: 850;
    letter-spacing: -2px;
}

/* ---------- Status ---------- */

.ss-status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(128,128,128,.10);
    font-size: 12px;
}

/* ---------- AI ---------- */

.ss-ai {
    border-radius: 26px;
    padding: 24px;
    border: 1px solid rgba(128,160,255,.20);
    background:
        linear-gradient(
            135deg,
            rgba(100,140,255,.13),
            rgba(128,128,128,.05)
        );
}

.ss-ai-title {
    font-size: 21px;
    font-weight: 800;
}

.ss-ai-label {
    font-size: 12px;
    opacity: .58;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ---------- Tabs ---------- */

button[data-baseweb="tab"] {
    border-radius: 14px;
}

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"] {
    background: rgba(128,128,128,.045);
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def fmt_money(value, digits=2):
    if value is None:
        return "—"

    try:
        if math.isnan(float(value)):
            return "—"

        return f"${float(value):,.{digits}f}"

    except Exception:
        return "—"


def fmt_percent(value, digits=2):
    if value is None:
        return "—"

    try:
        return f"{float(value) * 100:.{digits}f}%"

    except Exception:
        return "—"


def fmt_number(value):
    if value is None:
        return "—"

    try:
        value = float(value)

        if abs(value) >= 1e12:
            return f"{value / 1e12:.2f}T"

        if abs(value) >= 1e9:
            return f"{value / 1e9:.2f}B"

        if abs(value) >= 1e6:
            return f"{value / 1e6:.2f}M"

        return f"{value:,.0f}"

    except Exception:
        return "—"


def calculate_technicals(df: pd.DataFrame):
    data = df.copy()

    close = data["Close"]

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
        data["EMA12"] - data["EMA26"]
    )

    data["MACD_SIGNAL"] = (
        data["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["RSI"] = 100 - (
        100 / (1 + rs)
    )

    return data


def technical_summary(df):
    if df.empty:
        return {}

    latest = df.iloc[-1]

    price = latest.get("Close")
    sma20 = latest.get("SMA20")
    sma50 = latest.get("SMA50")
    sma200 = latest.get("SMA200")
    rsi = latest.get("RSI")

    trend = "NEUTRAL"

    if (
        pd.notna(sma50)
        and pd.notna(sma200)
    ):

        if price > sma50 > sma200:
            trend = "BULLISH"

        elif price < sma50 < sma200:
            trend = "BEARISH"

    momentum = "NEUTRAL"

    if pd.notna(rsi):

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


def build_ai_context(
    ticker,
    quote,
    fundamentals,
    technicals,
):
    return {
        "ticker": ticker,
        "quote": quote,
        "fundamentals": fundamentals,
        "technicals": technicals,
    }


# ============================================================
# SESSION STATE
# ============================================================

if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"

if "ai_report" not in st.session_state:
    st.session_state.ai_report = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ◈ Simon Stock"
    )

    st.caption(
        "V13.1 Foundation"
    )

    ticker_input = st.text_input(
        "股票代码",
        value=st.session_state.ticker,
        placeholder="AAPL / GOOGL / NVDA",
    )

    ticker = (
        ticker_input.strip().upper()
        if ticker_input
        else "AAPL"
    )

    st.session_state.ticker = ticker

    st.divider()

    period = st.selectbox(
        "历史周期",
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

    st.divider()

    st.markdown(
        "### AI Brain"
    )

    ai_status = ai_health_check()

    if ai_status["configured"]:

        st.success(
            f"AI Online · "
            f"{ai_status['provider']}"
        )

        st.caption(
            ai_status["model"]
        )

    else:

        st.warning(
            "AI 尚未连接"
        )

        st.caption(
            "请在部署平台配置 API Key"
        )

    st.divider()

    st.caption(
        "Market data: Yahoo Finance"
    )

    st.caption(
        "Research engine: Simon Stock AI"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="ss-header">

<div class="ss-brand">
◈ Simon Stock
</div>

<div class="ss-subtitle">
AI-Native US Equity Research Platform
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOAD
# ============================================================

try:

    with st.spinner(
        f"正在加载 {ticker}..."
    ):

        bundle = get_market_bundle(
            ticker,
            period=period,
            interval="1d",
        )

        history = bundle["history"]
        quote = bundle["quote"]
        fundamentals = bundle["fundamentals"]

except Exception as exc:

    st.error(
        f"无法加载 {ticker}：{exc}"
    )

    st.stop()


# ============================================================
# TECHNICALS
# ============================================================

technical_df = calculate_technicals(
    history
)

technicals = technical_summary(
    technical_df
)


# ============================================================
# HERO
# ============================================================

company_name = (
    fundamentals.get("name")
    or ticker
)

price = quote.get("price")
change = quote.get("change")
change_pct = quote.get(
    "change_percent"
)

change_sign = ""

if change is not None:

    if change > 0:
        change_sign = "+"

    elif change < 0:
        change_sign = ""


st.markdown(
    f"""
<div class="ss-hero">

<div class="ss-hero-symbol">
{ticker}
</div>

<div class="ss-hero-name">
{company_name}
</div>

<div class="ss-hero-price">
{fmt_money(price)}
</div>

<div class="ss-status">
{change_sign}{fmt_money(change)}
&nbsp;&nbsp;
{change_sign}{fmt_percent(change_pct)}
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# KPI ROW
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

with c1:

    st.metric(
        "Market Cap",
        fmt_number(
            fundamentals.get(
                "market_cap"
            )
        ),
    )

with c2:

    st.metric(
        "P/E",
        (
            f"{fundamentals.get('pe'):.2f}"
            if fundamentals.get("pe")
            else "—"
        ),
    )

with c3:

    st.metric(
        "Revenue Growth",
        fmt_percent(
            fundamentals.get(
                "revenue_growth"
            )
        ),
    )

with c4:

    st.metric(
        "ROE",
        fmt_percent(
            fundamentals.get("roe")
        ),
    )

with c5:

    st.metric(
        "RSI",
        (
            f"{technicals.get('rsi'):.1f}"
            if technicals.get("rsi") is not None
            else "—"
        ),
    )


# ============================================================
# MAIN TABS
# ============================================================

tab_market, tab_fundamental, tab_ai, tab_risk = st.tabs(
    [
        "📈 Market",
        "🏢 Fundamentals",
        "🧠 AI Research",
        "⚠️ Risk",
    ]
)


# ============================================================
# MARKET
# ============================================================

with tab_market:

    st.markdown(
        '<div class="ss-card-title">PRICE ACTION</div>',
        unsafe_allow_html=True,
    )

    chart_df = technical_df[
        [
            "Close",
            "SMA20",
            "SMA50",
            "SMA200",
        ]
    ].dropna(
        how="all"
    )

    st.line_chart(
        chart_df,
        height=430,
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Trend",
            technicals.get(
                "trend",
                "—",
            ),
        )

    with m2:

        st.metric(
            "Momentum",
            technicals.get(
                "momentum",
                "—",
            ),
        )

    with m3:

        st.metric(
            "Volume",
            fmt_number(
                quote.get("volume")
            ),
        )

    st.markdown(
        "### Technical Snapshot"
    )

    technical_table = pd.DataFrame(
        {
            "Indicator": [
                "Price",
                "SMA 20",
                "SMA 50",
                "SMA 200",
                "RSI",
            ],
            "Value": [
                fmt_money(price),
                fmt_money(
                    technicals.get(
                        "sma20"
                    )
                ),
                fmt_money(
                    technicals.get(
                        "sma50"
                    )
                ),
                fmt_money(
                    technicals.get(
                        "sma200"
                    )
                ),
                (
                    f"{technicals.get('rsi'):.2f}"
                    if technicals.get("rsi")
                    is not None
                    else "—"
                ),
            ],
        }
    )

    st.dataframe(
        technical_table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FUNDAMENTALS
# ============================================================

with tab_fundamental:

    st.markdown(
        "### Business & Valuation"
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        st.markdown(
            """
<div class="ss-card">
<div class="ss-card-title">VALUATION</div>
""",
            unsafe_allow_html=True,
        )

        st.write(
            f"**P/E:** "
            f"{fundamentals.get('pe') or '—'}"
        )

        st.write(
            f"**Forward P/E:** "
            f"{fundamentals.get('forward_pe') or '—'}"
        )

        st.write(
            f"**P/B:** "
            f"{fundamentals.get('price_to_book') or '—'}"
        )

        st.write(
            f"**EV/EBITDA:** "
            f"{fundamentals.get('ev_to_ebitda') or '—'}"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with f2:

        st.markdown(
            """
<div class="ss-card">
<div class="ss-card-title">GROWTH & PROFITABILITY</div>
""",
            unsafe_allow_html=True,
        )

        st.write(
            f"**Revenue Growth:** "
            f"{fmt_percent(fundamentals.get('revenue_growth'))}"
        )

        st.write(
            f"**Earnings Growth:** "
            f"{fmt_percent(fundamentals.get('earnings_growth'))}"
        )

        st.write(
            f"**Gross Margin:** "
            f"{fmt_percent(fundamentals.get('gross_margin'))}"
        )

        st.write(
            f"**Operating Margin:** "
            f"{fmt_percent(fundamentals.get('operating_margin'))}"
        )

        st.write(
            f"**ROE:** "
            f"{fmt_percent(fundamentals.get('roe'))}"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with f3:

        st.markdown(
            """
<div class="ss-card">
<div class="ss-card-title">BALANCE SHEET</div>
""",
            unsafe_allow_html=True,
        )

        st.write(
            f"**Cash:** "
            f"{fmt_number(fundamentals.get('total_cash'))}"
        )

        st.write(
            f"**Debt:** "
            f"{fmt_number(fundamentals.get('total_debt'))}"
        )

        st.write(
            f"**Current Ratio:** "
            f"{fundamentals.get('current_ratio') or '—'}"
        )

        st.write(
            f"**Debt / Equity:** "
            f"{fundamentals.get('debt_to_equity') or '—'}"
        )

        st.write(
            f"**Free Cash Flow:** "
            f"{fmt_number(fundamentals.get('free_cash_flow'))}"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "### Company Profile"
    )

    p1, p2 = st.columns(2)

    with p1:

        st.write(
            f"**Sector:** "
            f"{fundamentals.get('sector') or '—'}"
        )

        st.write(
            f"**Industry:** "
            f"{fundamentals.get('industry') or '—'}"
        )

    with p2:

        st.write(
            f"**Country:** "
            f"{fundamentals.get('country') or '—'}"
        )

        st.write(
            f"**Website:** "
            f"{fundamentals.get('website') or '—'}"
        )


# ============================================================
# AI RESEARCH
# ============================================================

with tab_ai:

    ai_status = ai_health_check()

    if not ai_status["configured"]:

        st.markdown(
            """
<div class="ss-ai">

<div class="ss-ai-title">
🧠 Simon Stock AI
</div>

<br>

<div>
AI 大脑目前没有连接。
</div>

<br>

请先配置 AI API Key。
配置完成后，这里会启动：
</div>
""",
            unsafe_allow_html=True,
        )

        st.info(
            """
V13.1 AI Pipeline：

Value Agent
→ Business Agent
→ First Principles Agent
→ Event Agent
→ Bull / Bear Debate
→ Investment Committee
"""
        )

    else:

        st.markdown(
            f"""
<div class="ss-ai">

<div class="ss-ai-label">
AI Research Engine
</div>

<div class="ss-ai-title">
Investment Intelligence
</div>

<div>
Provider: {ai_status["provider"]}
&nbsp; · &nbsp;
Model: {ai_status["model"]}
</div>

</div>
""",
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "🧠 启动深度 AI 研究",
            use_container_width=True,
            type="primary",
        ):

            ai_context = build_ai_context(
                ticker,
                quote,
                fundamentals,
                technicals,
            )

            with st.spinner(
                "AI Research Committee 正在分析..."
            ):

                try:

                    report = run_full_ai_research(
                        ticker,
                        ai_context,
                    )

                    st.session_state.ai_report = report

                except Exception as exc:

                    st.error(
                        f"AI 分析失败：{exc}"
                    )

        report = st.session_state.ai_report

        if report:

            st.divider()

            st.markdown(
                "## Investment Committee"
            )

            committee = report.get(
                "committee",
                {},
            )

            if committee.get("success"):

                st.markdown(
                    committee.get(
                        "content",
                        "暂无结果",
                    )
                )

            else:

                st.warning(
                    committee.get(
                        "error",
                        "AI Committee unavailable.",
                    )
                )

            st.divider()

            st.markdown(
                "## Research Agents"
            )

            for agent in report.get(
                "agents",
                [],
            ):

                with st.expander(
                    agent.get(
                        "agent",
                        "Agent",
                    ).replace(
                        "_",
                        " "
                    ).upper()
                ):

                    st.markdown(
                        agent.get(
                            "conclusion",
                            "暂无分析",
                        )
                    )

            debate = report.get(
                "debate",
                {},
            )

            if debate:

                st.divider()

                st.markdown(
                    "## Bull / Bear Debate"
                )

                if debate.get(
                    "success"
                ):

                    st.markdown(
                        debate.get(
                            "content",
                            "",
                        )
                    )

                else:

                    st.warning(
                        debate.get(
                            "error",
                            "Debate unavailable.",
                        )
                    )


# ============================================================
# RISK
# ============================================================

with tab_risk:

    st.markdown(
        "### Risk Dashboard"
    )

    risk_score = 50

    if fundamentals.get(
        "debt_to_equity"
    ):

        debt = float(
            fundamentals[
                "debt_to_equity"
            ]
        )

        if debt > 150:
            risk_score += 20

        elif debt < 50:
            risk_score -= 10

    if technicals.get(
        "trend"
    ) == "BEARISH":

        risk_score += 15

    if technicals.get(
        "momentum"
    ) == "OVERBOUGHT":

        risk_score += 10

    risk_score = max(
        0,
        min(
            100,
            risk_score,
        )
    )

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Risk Score",
            f"{risk_score}/100",
        )

    with r2:

        st.metric(
            "Trend Risk",
            technicals.get(
                "trend",
                "—",
            ),
        )

    with r3:

        st.metric(
            "Balance Sheet",
            (
                "Watch"
                if fundamentals.get(
                    "debt_to_equity"
                )
                and fundamentals[
                    "debt_to_equity"
                ] > 150
                else "Normal"
            ),
        )

    st.info(
        """
风险评分是研究辅助指标，不是预测价格的模型。
真正的风险判断需要结合估值、商业质量、
宏观环境、仓位和投资期限。
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"Simon Stock V13.1 Foundation · "
    f"Last analysis: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)

st.caption(
    "Research tool only · Not financial advice"
)
