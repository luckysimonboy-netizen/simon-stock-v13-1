"""
Simon Stock V13.1 Foundation
Theme & UI System

Design direction:
- System-aware Light / Dark
- Liquid Glass inspired surfaces
- OxygenOS / ColorOS inspired simplicity
- Financial-data-first readability
"""

from __future__ import annotations

import streamlit as st


# ============================================================
# Theme Configuration
# ============================================================

THEMES = {
    "light": {
        "background": "#F5F7FA",
        "surface": "rgba(255,255,255,0.72)",
        "surface_strong": "rgba(255,255,255,0.90)",
        "border": "rgba(0,0,0,0.07)",
        "text": "#111318",
        "text_secondary": "#6B7280",
        "accent": "#1677FF",
        "accent_soft": "rgba(22,119,255,0.10)",
        "positive": "#16834B",
        "negative": "#D94A4A",
        "warning": "#B77900",
        "shadow": "0 8px 30px rgba(20,30,50,0.07)",
    },
    "dark": {
        "background": "#090B0F",
        "surface": "rgba(24,27,33,0.72)",
        "surface_strong": "rgba(29,33,40,0.90)",
        "border": "rgba(255,255,255,0.08)",
        "text": "#F5F7FA",
        "text_secondary": "#9CA3AF",
        "accent": "#4B9BFF",
        "accent_soft": "rgba(75,155,255,0.12)",
        "positive": "#36C985",
        "negative": "#FF6868",
        "warning": "#E3AE43",
        "shadow": "0 10px 35px rgba(0,0,0,0.30)",
    },
}


# ============================================================
# Session State
# ============================================================

def init_theme() -> None:
    """Initialize theme state."""

    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "system"


def get_theme_mode() -> str:
    """Return current theme preference."""

    init_theme()

    return st.session_state.get(
        "theme_mode",
        "system"
    )


def set_theme_mode(mode: str) -> None:
    """Set theme preference."""

    if mode not in {
        "system",
        "light",
        "dark",
    }:
        mode = "system"

    st.session_state.theme_mode = mode


# ============================================================
# Theme CSS
# ============================================================

def inject_theme_css() -> None:
    """
    Inject the global V13.1 design system.

    System preference is respected through CSS media queries.
    """

    css = r"""
<style>

/* ==========================================================
   ROOT
   ========================================================== */

:root {
    --ss-radius: 22px;
    --ss-radius-small: 14px;
    --ss-transition: 180ms cubic-bezier(.2,.8,.2,1);
}


/* ==========================================================
   GLOBAL
   ========================================================== */

html, body, [class*="css"] {
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "SF Pro Display",
        "SF Pro Text",
        "Segoe UI",
        Roboto,
        Helvetica,
        Arial,
        sans-serif;
}

.stApp {
    background: #F5F7FA;
    color: #111318;
    transition:
        background-color 180ms ease,
        color 180ms ease;
}


/* ==========================================================
   SYSTEM DARK MODE
   ========================================================== */

@media (prefers-color-scheme: dark) {

    .stApp {
        background: #090B0F;
        color: #F5F7FA;
    }

    section[data-testid="stSidebar"] {
        background: rgba(14,16,21,0.86);
        border-right-color: rgba(255,255,255,0.07);
    }

    div[data-testid="stMetric"] {
        background: rgba(25,28,34,0.72);
        border-color: rgba(255,255,255,0.08);
    }

    .ss-glass {
        background: rgba(24,27,33,0.68);
        border-color: rgba(255,255,255,0.08);
        box-shadow:
            0 10px 35px rgba(0,0,0,0.30);
    }

    .ss-secondary {
        color: #9CA3AF;
    }

    .ss-title {
        color: #F5F7FA;
    }
}


/* ==========================================================
   MAIN CONTAINER
   ========================================================== */

.block-container {
    max-width: 1500px;
    padding-top: 1.25rem;
    padding-bottom: 4rem;
}


/* ==========================================================
   HEADINGS
   ========================================================== */

h1, h2, h3 {
    letter-spacing: -0.025em;
}

h1 {
    font-weight: 750;
}

h2 {
    font-weight: 700;
}

h3 {
    font-weight: 650;
}


/* ==========================================================
   GLASS CARD
   ========================================================== */

.ss-glass {
    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.78),
            rgba(255,255,255,0.58)
        );

    border: 1px solid rgba(0,0,0,0.06);

    border-radius: var(--ss-radius);

    padding: 20px;

    box-shadow:
        0 8px 30px rgba(20,30,50,0.07);

    backdrop-filter: blur(24px) saturate(135%);
    -webkit-backdrop-filter:
        blur(24px) saturate(135%);

    transition:
        transform var(--ss-transition),
        box-shadow var(--ss-transition),
        border-color var(--ss-transition);
}

.ss-glass:hover {
    transform: translateY(-1px);

    box-shadow:
        0 14px 38px rgba(20,30,50,0.10);
}


/* ==========================================================
   SMALL GLASS
   ========================================================== */

.ss-glass-small {
    background:
        rgba(255,255,255,0.62);

    border:
        1px solid rgba(0,0,0,0.055);

    border-radius:
        var(--ss-radius-small);

    padding:
        14px;

    backdrop-filter:
        blur(20px);

    -webkit-backdrop-filter:
        blur(20px);
}


/* ==========================================================
   TITLE
   ========================================================== */

.ss-title {
    font-size: 1.55rem;
    font-weight: 760;
    letter-spacing: -0.035em;
    margin-bottom: 4px;
}

.ss-subtitle {
    font-size: 0.92rem;
    color: #6B7280;
}


/* ==========================================================
   METRIC CARDS
   ========================================================== */

div[data-testid="stMetric"] {

    background:
        rgba(255,255,255,0.68);

    border:
        1px solid rgba(0,0,0,0.055);

    border-radius:
        18px;

    padding:
        14px 16px;

    backdrop-filter:
        blur(18px);

    -webkit-backdrop-filter:
        blur(18px);

    box-shadow:
        0 5px 20px rgba(20,30,50,0.045);

    transition:
        transform var(--ss-transition),
        box-shadow var(--ss-transition);
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-1px);

    box-shadow:
        0 10px 25px rgba(20,30,50,0.08);
}


/* ==========================================================
   BUTTONS
   ========================================================== */

.stButton > button {

    border-radius:
        14px;

    border:
        1px solid rgba(0,0,0,0.07);

    min-height:
        42px;

    font-weight:
        620;

    transition:
        transform var(--ss-transition),
        box-shadow var(--ss-transition),
        background-color var(--ss-transition);
}

.stButton > button:hover {

    transform:
        translateY(-1px);

    box-shadow:
        0 7px 20px rgba(0,0,0,0.08);
}

.stButton > button:active {

    transform:
        scale(0.985);
}


/* ==========================================================
   INPUTS
   ========================================================== */

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {

    border-radius:
        14px !important;

    transition:
        border-color var(--ss-transition),
        box-shadow var(--ss-transition);
}


/* ==========================================================
   TABS
   ========================================================== */

button[data-baseweb="tab"] {

    font-weight:
        600;

    border-radius:
        12px;

    transition:
        background-color var(--ss-transition);
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {

    background:
        rgba(248,249,251,0.82);

    border-right:
        1px solid rgba(0,0,0,0.055);

    backdrop-filter:
        blur(24px);

    -webkit-backdrop-filter:
        blur(24px);
}


/* ==========================================================
   DATAFRAME
   ========================================================== */

div[data-testid="stDataFrame"] {

    border-radius:
        18px;

    overflow:
        hidden;

    border:
        1px solid rgba(0,0,0,0.06);
}


/* ==========================================================
   ALERTS
   ========================================================== */

div[data-testid="stAlert"] {

    border-radius:
        16px;
}


/* ==========================================================
   DIVIDER
   ========================================================== */

hr {

    border:
        none;

    border-top:
        1px solid rgba(0,0,0,0.07);

    margin:
        1.2rem 0;
}


/* ==========================================================
   SCROLLBAR
   ========================================================== */

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-thumb {
    border-radius: 999px;
    background: rgba(120,130,145,0.35);
}

::-webkit-scrollbar-track {
    background: transparent;
}


/* ==========================================================
   MOBILE
   ========================================================== */

@media (max-width: 768px) {

    .block-container {
        padding-left: 0.85rem;
        padding-right: 0.85rem;
    }

    .ss-glass {
        padding: 16px;
        border-radius: 18px;
    }

    .ss-title {
        font-size: 1.3rem;
    }
}

</style>
"""

    st.markdown(
        css,
        unsafe_allow_html=True
    )


# ============================================================
# Theme Selector
# ============================================================

def render_theme_selector() -> str:
    """
    Render user-facing theme selector.

    Options:
    System
    Light
    Dark
    """

    init_theme()

    current = get_theme_mode()

    labels = {
        "system": "System",
        "light": "Light",
        "dark": "Dark",
    }

    options = list(labels.keys())

    selected = st.selectbox(
        "Appearance",
        options=options,
        index=options.index(current),
        format_func=lambda x: labels[x],
        key="appearance_selector",
    )

    if selected != current:
        set_theme_mode(selected)
        st.rerun()

    return selected


# ============================================================
# UI Helpers
# ============================================================

def glass_card(
    title: str,
    value: str,
    subtitle: str = "",
) -> None:
    """Render a lightweight glass information card."""

    subtitle_html = ""

    if subtitle:
        subtitle_html = (
            f'<div class="ss-subtitle">'
            f'{subtitle}'
            f'</div>'
        )

    html = f"""
    <div class="ss-glass">
        <div class="ss-subtitle">
            {title}
        </div>

        <div style="
            font-size:1.65rem;
            font-weight:760;
            letter-spacing:-0.03em;
            margin-top:5px;
        ">
            {value}
        </div>

        {subtitle_html}
    </div>
    """

    st.markdown(
        html,
        unsafe_allow_html=True
    )


def section_header(
    title: str,
    subtitle: str = "",
) -> None:
    """Render a consistent section header."""

    html = f"""
    <div style="margin: 12px 0 16px 0;">
        <div class="ss-title">
            {title}
        </div>
        {
            f'<div class="ss-subtitle">{subtitle}</div>'
            if subtitle else ""
        }
    </div>
    """

    st.markdown(
        html,
        unsafe_allow_html=True
    )


def score_badge(
    score: float,
    label: str = "Score",
) -> None:
    """Render a compact score badge."""

    score = max(
        0,
        min(100, float(score))
    )

    if score >= 75:
        symbol = "●"
    elif score >= 55:
        symbol = "◐"
    else:
        symbol = "○"

    html = f"""
    <div class="ss-glass-small"
         style="display:inline-block;">

        <span style="font-weight:700;">
            {symbol} {label}
        </span>

        <span style="
            margin-left:8px;
            font-weight:800;
        ">
            {score:.1f}
        </span>

    </div>
    """

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# Public Theme API
# ============================================================

def setup_theme() -> None:
    """
    One-call theme initialization.

    Main app should simply call:

        setup_theme()
    """

    init_theme()
    inject_theme_css()
