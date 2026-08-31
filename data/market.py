"""
Simon Stock V13.1 Foundation
Market Data Layer

Responsibilities:
- Yahoo Finance market data
- Historical OHLCV
- Company information
- Financial statements
- Valuation metrics
- Dividends / splits
- Earnings
- Analyst data
- Safe caching
- Error handling

Important:
This module is a data adapter, not an investment recommendation engine.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# Configuration
# ============================================================

DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"

VALID_PERIODS = {
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "10y",
    "max",
}

VALID_INTERVALS = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}


# ============================================================
# Helpers
# ============================================================

def normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker symbol.

    Examples:
        apple -> AAPL
         aapl  -> AAPL
    """

    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    symbol = str(ticker).strip().upper()

    if not symbol:
        raise ValueError("Ticker cannot be empty.")

    return symbol


def safe_number(value: Any) -> Optional[float]:
    """Convert values safely to float."""

    try:
        if value is None:
            return None

        if pd.isna(value):
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def clean_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove values that cannot be safely serialized.
    """

    result = {}

    for key, value in data.items():

        if isinstance(value, (pd.Timestamp,)):
            result[key] = value.isoformat()
            continue

        if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
            result[key] = None
            continue

        result[key] = value

    return result


# ============================================================
# Historical Market Data
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def get_history(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """
    Download historical OHLCV data.

    Cache:
        5 minutes

    Returns:
        DataFrame
    """

    symbol = normalize_ticker(ticker)

    if period not in VALID_PERIODS:
        period = DEFAULT_PERIOD

    if interval not in VALID_INTERVALS:
        interval = DEFAULT_INTERVAL

    try:

        data = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Unable to download market data for {symbol}: {exc}"
        ) from exc

    if data is None or data.empty:
        raise RuntimeError(
            f"No market data returned for {symbol}."
        )

    # Flatten MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):

        flattened = []

        for column in data.columns:

            parts = [
                str(part)
                for part in column
                if str(part) not in {"", "None"}
            ]

            flattened.append("_".join(parts))

        data.columns = flattened

        # Rename based on first component
        rename = {}

        for column in data.columns:

            lower = column.lower()

            if lower.startswith("open"):
                rename[column] = "Open"

            elif lower.startswith("high"):
                rename[column] = "High"

            elif lower.startswith("low"):
                rename[column] = "Low"

            elif lower.startswith("close"):
                rename[column] = "Close"

            elif lower.startswith("adj close"):
                rename[column] = "Adj Close"

            elif lower.startswith("volume"):
                rename[column] = "Volume"

        data = data.rename(columns=rename)

    # Standard columns
    expected_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    for column in expected_columns:

        if column not in data.columns:

            if column == "Adj Close" and "Close" in data.columns:
                data[column] = data["Close"]

            elif column == "Volume":
                data[column] = 0

            else:
                data[column] = None

    # Convert numeric columns
    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.sort_index()

    data = data.dropna(
        subset=["Close"]
    )

    return data


# ============================================================
# Current Quote
# ============================================================

@st.cache_data(
    ttl=60,
    show_spinner=False
)
def get_quote(ticker: str) -> Dict[str, Any]:
    """
    Get current quote information.

    Cache:
        60 seconds
    """

    symbol = normalize_ticker(ticker)

    try:

        history = get_history(
            symbol,
            period="5d",
            interval="1d"
        )

        if history.empty:
            raise RuntimeError(
                f"No quote data for {symbol}"
            )

        latest = history.iloc[-1]

        close = safe_number(
            latest.get("Close")
        )

        previous_close = None

        if len(history) >= 2:

            previous_close = safe_number(
                history.iloc[-2].get("Close")
            )

        change = None
        change_percent = None

        if close is not None and previous_close not in {
            None,
            0
        }:

            change = close - previous_close

            change_percent = (
                change / previous_close
            )

        return {
            "ticker": symbol,
            "price": close,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change_percent,
            "open": safe_number(
                latest.get("Open")
            ),
            "high": safe_number(
                latest.get("High")
            ),
            "low": safe_number(
                latest.get("Low")
            ),
            "volume": safe_number(
                latest.get("Volume")
            ),
            "timestamp": str(
                history.index[-1]
            ),
        }

    except Exception as exc:

        raise RuntimeError(
            f"Unable to retrieve quote for {symbol}: {exc}"
        ) from exc


# ============================================================
# Company Information
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def get_company_info(
    ticker: str
) -> Dict[str, Any]:
    """
    Retrieve company profile and basic information.

    Cache:
        1 hour
    """

    symbol = normalize_ticker(ticker)

    try:

        stock = yf.Ticker(symbol)

        info = stock.info

        if not isinstance(info, dict):
            return {}

        return clean_dict(info)

    except Exception:

        # Some Yahoo endpoints can fail independently.
        # Return empty dictionary rather than crashing the UI.
        return {}


# ============================================================
# Fast Company Snapshot
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def get_fundamental_snapshot(
    ticker: str
) -> Dict[str, Any]:
    """
    Extract the most useful fundamental fields
    from Yahoo Finance.

    This is intentionally normalized so the AI
    layer does not need to understand Yahoo's
    raw field names.
    """

    symbol = normalize_ticker(ticker)

    info = get_company_info(symbol)

    def first_value(*keys):

        for key in keys:

            value = info.get(key)

            if value is not None:
                return value

        return None

    return {
        "ticker": symbol,

        # Company
        "name": first_value(
            "longName",
            "shortName"
        ),

        "sector": first_value(
            "sector"
        ),

        "industry": first_value(
            "industry"
        ),

        "country": first_value(
            "country"
        ),

        "website": first_value(
            "website"
        ),

        # Market
        "market_cap": first_value(
            "marketCap"
        ),

        "enterprise_value": first_value(
            "enterpriseValue"
        ),

        # Valuation
        "pe": first_value(
            "trailingPE"
        ),

        "forward_pe": first_value(
            "forwardPE"
        ),

        "peg": first_value(
            "pegRatio"
        ),

        "price_to_book": first_value(
            "priceToBook"
        ),

        "price_to_sales": first_value(
            "priceToSalesTrailing12Months"
        ),

        "ev_to_ebitda": first_value(
            "enterpriseToEbitda"
        ),

        # Growth
        "revenue_growth": first_value(
            "revenueGrowth"
        ),

        "earnings_growth": first_value(
            "earningsGrowth"
        ),

        "earnings_quarterly_growth": first_value(
            "earningsQuarterlyGrowth"
        ),

        # Profitability
        "profit_margin": first_value(
            "profitMargins"
        ),

        "operating_margin": first_value(
            "operatingMargins"
        ),

        "gross_margin": first_value(
            "grossMargins"
        ),

        "roe": first_value(
            "returnOnEquity"
        ),

        "roa": first_value(
            "returnOnAssets"
        ),

        # Cash flow
        "free_cash_flow": first_value(
            "freeCashflow"
        ),

        "operating_cash_flow": first_value(
            "operatingCashflow"
        ),

        # Balance sheet
        "total_cash": first_value(
            "totalCash"
        ),

        "total_debt": first_value(
            "totalDebt"
        ),

        "debt_to_equity": first_value(
            "debtToEquity"
        ),

        "current_ratio": first_value(
            "currentRatio"
        ),

        # Per share
        "eps": first_value(
            "trailingEps"
        ),

        "forward_eps": first_value(
            "forwardEps"
        ),

        # Dividend
        "dividend_rate": first_value(
            "dividendRate"
        ),

        "dividend_yield": first_value(
            "dividendYield"
        ),

        "payout_ratio": first_value(
            "payoutRatio"
        ),

        # Analyst
        "target_mean": first_value(
            "targetMeanPrice"
        ),

        "target_low": first_value(
            "targetLowPrice"
        ),

        "target_high": first_value(
            "targetHighPrice"
        ),

        "recommendation": first_value(
            "recommendationKey"
        ),

        "analyst_count": first_value(
            "numberOfAnalystOpinions"
        ),

        # Shares
        "shares_outstanding": first_value(
            "sharesOutstanding"
        ),

        "float_shares": first_value(
            "floatShares"
        ),
    }


# ============================================================
# Financial Statements
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def get_income_statement(
    ticker: str
) -> pd.DataFrame:
    """Return annual income statement."""

    symbol = normalize_ticker(ticker)

    try:

        stock = yf.Ticker(symbol)

        data = stock.financials

        if data is None:
            return pd.DataFrame()

        return data

    except Exception:

        return pd.DataFrame()


@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def get_balance_sheet(
    ticker: str
) -> pd.DataFrame:
    """Return annual balance sheet."""

    symbol = normalize_ticker(ticker)

    try:

        stock = yf.Ticker(symbol)

        data = stock.balance_sheet

        if data is None:
            return pd.DataFrame()

        return data

    except Exception:

        return pd.DataFrame()


@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def get_cash_flow(
    ticker: str
) -> pd.DataFrame:
    """Return annual cash-flow statement."""

    symbol = normalize_ticker(ticker)

    try:

        stock = yf.Ticker(symbol)

        data = stock.cashflow

        if data is None:
            return pd.DataFrame()

        return data

    except Exception:

        return pd.DataFrame()


# ============================================================
# Quarterly Financials
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def get_quarterly_financials(
    ticker: str
) -> Dict[str, pd.DataFrame]:
    """
    Return quarterly financial statements.
    """

    symbol = normalize_ticker(ticker)

    try:

        stock = yf.Ticker(symbol)

        return {
            "income_statement":
                stock.quarterly_financials,

            "balance_sheet":
                stock.quarterly_balance_sheet,

            "cash_flow":
                stock.quarterly_cashflow,
        }

    except Exception:

        return {
            "income_statement":
                pd.DataFrame(),

            "balance_sheet":
                pd.DataFrame(),

            "cash_flow":
                pd.DataFrame(),
        }


# ============================================================
# Dividends & Splits
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def get_actions(
    ticker: str
) -> pd.DataFrame:
    """
    Return dividends and stock splits.
    """

    symbol = normalize_ticker(ticker)

    try:

        stock = yf.Ticker(symbol)

        actions = stock.actions

        if actions is None:
            return pd.DataFrame()

        return actions

    except Exception:

        return pd.DataFrame()


@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def get_dividends(
    ticker: str
) -> pd.Series:
    """Return dividend history."""

    symbol = normalize_ticker(ticker)

    try:

        stock = yf.Ticker(symbol)

        dividends = stock.dividends

        if dividends is None:
            return pd.Series(dtype=float)

        return dividends

    except Exception:

        return pd.Series(dtype=float)


# ============================================================
# Earnings
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def get_earnings_history(
    ticker: str
) -> pd.DataFrame:
    """Return earnings history when available."""

    symbol = normalize_ticker(ticker)

    try:

        stock = yf.Ticker(symbol)

        earnings = stock.earnings_dates

        if earnings is None:
            return pd.DataFrame()

        return earnings

    except Exception:

        return pd.DataFrame()


# ============================================================
# Analyst Data
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def get_analyst_targets(
    ticker: str
) -> Dict[str, Any]:
    """
    Return analyst target-price information.
    """

    symbol = normalize_ticker(ticker)

    info = get_company_info(symbol)

    return {
        "ticker": symbol,
        "target_low": safe_number(
            info.get("targetLowPrice")
        ),
        "target_mean": safe_number(
            info.get("targetMeanPrice")
        ),
        "target_high": safe_number(
            info.get("targetHighPrice")
        ),
        "recommendation": info.get(
            "recommendationKey"
        ),
        "recommendation_mean": safe_number(
            info.get("recommendationMean")
        ),
        "analyst_count": safe_number(
            info.get("numberOfAnalystOpinions")
        ),
    }


# ============================================================
# Market Data Bundle
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def get_market_bundle(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> Dict[str, Any]:
    """
    Fetch the main market-data bundle used by
    the V13.1 analysis pipeline.
    """

    symbol = normalize_ticker(ticker)

    history = get_history(
        symbol,
        period=period,
        interval=interval
    )

    quote = get_quote(symbol)

    fundamentals = get_fundamental_snapshot(
        symbol
    )

    return {
        "ticker": symbol,
        "quote": quote,
        "history": history,
        "fundamentals": fundamentals,
    }


# ============================================================
# Health Check
# ============================================================

def data_health_check(
    ticker: str
) -> Dict[str, Any]:
    """
    Check whether the main data endpoints are working.
    """

    symbol = normalize_ticker(ticker)

    result = {
        "ticker": symbol,
        "market_data": False,
        "quote": False,
        "fundamentals": False,
        "error": None,
    }

    try:

        history = get_history(
            symbol,
            period="1mo",
            interval="1d"
        )

        result["market_data"] = (
            not history.empty
        )

        quote = get_quote(symbol)

        result["quote"] = (
            quote.get("price") is not None
        )

        fundamentals = get_fundamental_snapshot(
            symbol
        )

        result["fundamentals"] = (
            len(fundamentals) > 0
        )

    except Exception as exc:

        result["error"] = str(exc)

    result["healthy"] = all([
        result["market_data"],
        result["quote"],
    ])

    return result


# ============================================================
# Cache Control
# ============================================================

def clear_market_cache() -> None:
    """
    Clear Streamlit market-data cache.
    """

    try:
        st.cache_data.clear()
    except Exception:
        pass
