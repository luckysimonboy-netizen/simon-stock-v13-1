"""
Simon Stock V13.1 Foundation
Core Analysis Engine

负责：
- 标准化股票数据
- 技术指标计算
- 综合评分
- 风险指标
- 基础信号
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ============================================================
# Data Models
# ============================================================

@dataclass
class AnalysisResult:
    ticker: str
    price: float
    trend_score: float
    momentum_score: float
    quality_score: float
    valuation_score: float
    risk_score: float
    composite_score: float
    signal: str
    confidence: float
    risk_level: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Utility
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Limit a score to a fixed range."""
    return max(low, min(high, value))


# ============================================================
# Data Normalization
# ============================================================

def normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV dataframe.

    Supports:
    Open, High, Low, Close, Adj Close, Volume
    """

    if data is None or data.empty:
        return pd.DataFrame()

    df = data.copy()

    # Flatten MultiIndex columns if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if str(x) != ""])
            for col in df.columns
        ]

    # Normalize column names
    rename_map = {}

    for column in df.columns:
        normalized = str(column).strip().lower()

        if normalized.startswith("open"):
            rename_map[column] = "Open"
        elif normalized.startswith("high"):
            rename_map[column] = "High"
        elif normalized.startswith("low"):
            rename_map[column] = "Low"
        elif normalized.startswith("close"):
            rename_map[column] = "Close"
        elif normalized.startswith("adj close"):
            rename_map[column] = "Adj Close"
        elif normalized.startswith("volume"):
            rename_map[column] = "Volume"

    df = df.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close", "Volume"]

    for column in required:
        if column not in df.columns:
            df[column] = np.nan

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_index()
    df = df.dropna(subset=["Close"])

    return df


# ============================================================
# Technical Indicators
# ============================================================

def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add the core technical indicators."""

    df = normalize_ohlcv(data)

    if df.empty:
        return df

    close = df["Close"]

    # Moving averages
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA100"] = close.rolling(100).mean()
    df["SMA200"] = close.rolling(200).mean()

    # EMA
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_Hist"] = (
        df["MACD"] - df["MACD_Signal"]
    )

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI14"] = 100 - (100 / (1 + rs))

    # ATR
    previous_close = close.shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - previous_close).abs()
    tr3 = (df["Low"] - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR14"] = true_range.rolling(14).mean()

    # Daily returns
    df["Return1D"] = close.pct_change()
    df["Return5D"] = close.pct_change(5)
    df["Return20D"] = close.pct_change(20)
    df["Return60D"] = close.pct_change(60)
    df["Return252D"] = close.pct_change(252)

    # Volatility
    df["Volatility20D"] = (
        df["Return1D"]
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    # Volume moving average
    df["VolumeSMA20"] = df["Volume"].rolling(20).mean()

    # Relative volume
    df["RelativeVolume"] = (
        df["Volume"] /
        df["VolumeSMA20"].replace(0, np.nan)
    )

    # 52-week high / low
    df["52W_High"] = close.rolling(252).max()
    df["52W_Low"] = close.rolling(252).min()

    return df


# ============================================================
# Trend Engine
# ============================================================

def calculate_trend_score(df: pd.DataFrame) -> float:
    """Calculate a 0-100 trend score."""

    if df.empty:
        return 50.0

    row = df.iloc[-1]

    score = 50.0

    close = safe_float(row.get("Close"))
    sma20 = safe_float(row.get("SMA20"))
    sma50 = safe_float(row.get("SMA50"))
    sma200 = safe_float(row.get("SMA200"))

    if close > sma20:
        score += 8

    if close > sma50:
        score += 10

    if close > sma200:
        score += 12

    if sma20 > sma50:
        score += 8

    if sma50 > sma200:
        score += 8

    if close < sma20:
        score -= 6

    if close < sma50:
        score -= 8

    if close < sma200:
        score -= 12

    return clamp(score)


# ============================================================
# Momentum Engine
# ============================================================

def calculate_momentum_score(df: pd.DataFrame) -> float:
    """Calculate momentum score."""

    if df.empty:
        return 50.0

    row = df.iloc[-1]

    score = 50.0

    rsi = safe_float(row.get("RSI14"), 50)
    macd = safe_float(row.get("MACD"))
    macd_signal = safe_float(row.get("MACD_Signal"))

    return20 = safe_float(row.get("Return20D"))
    return60 = safe_float(row.get("Return60D"))

    # RSI
    if 50 <= rsi <= 70:
        score += 12
    elif 40 <= rsi < 50:
        score -= 2
    elif rsi > 75:
        score -= 8
    elif rsi < 30:
        score += 4

    # MACD
    if macd > macd_signal:
        score += 10
    else:
        score -= 8

    # Price momentum
    if return20 > 0:
        score += 8
    else:
        score -= 6

    if return60 > 0:
        score += 8
    else:
        score -= 6

    return clamp(score)


# ============================================================
# Risk Engine
# ============================================================

def calculate_risk_score(df: pd.DataFrame) -> float:
    """
    Calculate risk score.

    Higher score = lower risk.
    """

    if df.empty:
        return 50.0

    returns = df["Return1D"].dropna()

    if returns.empty:
        return 50.0

    volatility = safe_float(
        returns.std() * np.sqrt(252),
        0.30
    )

    close = df["Close"]

    rolling_max = close.cummax()

    drawdown = (
        close / rolling_max - 1
    )

    max_drawdown = abs(
        safe_float(drawdown.min(), 0)
    )

    score = 100.0

    # Volatility penalty
    if volatility > 0.60:
        score -= 35
    elif volatility > 0.45:
        score -= 25
    elif volatility > 0.30:
        score -= 15
    elif volatility > 0.20:
        score -= 8

    # Drawdown penalty
    if max_drawdown > 0.60:
        score -= 35
    elif max_drawdown > 0.40:
        score -= 25
    elif max_drawdown > 0.25:
        score -= 15
    elif max_drawdown > 0.15:
        score -= 8

    return clamp(score)


# ============================================================
# Valuation / Quality Placeholder
# ============================================================

def calculate_valuation_score(
    fundamentals: Optional[Dict[str, Any]] = None
) -> float:
    """
    Foundation valuation engine.

    V13.1 will later connect:
    - P/E
    - Forward P/E
    - PEG
    - EV/EBITDA
    - FCF yield
    - historical valuation percentile
    - DCF
    """

    if not fundamentals:
        return 50.0

    score = 50.0

    pe = safe_float(
        fundamentals.get("pe")
    )

    growth = safe_float(
        fundamentals.get("growth")
    )

    if pe > 0:
        if pe < 15:
            score += 15
        elif pe < 25:
            score += 8
        elif pe > 45:
            score -= 15

    if growth > 0.20:
        score += 10

    return clamp(score)


def calculate_quality_score(
    fundamentals: Optional[Dict[str, Any]] = None
) -> float:
    """Foundation business quality score."""

    if not fundamentals:
        return 50.0

    score = 50.0

    roe = safe_float(
        fundamentals.get("roe")
    )

    margin = safe_float(
        fundamentals.get("margin")
    )

    revenue_growth = safe_float(
        fundamentals.get("revenue_growth")
    )

    if roe > 0.20:
        score += 15
    elif roe > 0.10:
        score += 8

    if margin > 0.20:
        score += 15
    elif margin > 0.10:
        score += 8

    if revenue_growth > 0.15:
        score += 10

    return clamp(score)


# ============================================================
# Composite Engine
# ============================================================

def calculate_composite_score(
    trend_score: float,
    momentum_score: float,
    quality_score: float,
    valuation_score: float,
    risk_score: float,
) -> float:
    """
    Combine multiple analytical dimensions.

    This is deliberately modular so V13.x can later
    introduce adaptive AI weighting.
    """

    weights = {
        "trend": 0.20,
        "momentum": 0.15,
        "quality": 0.25,
        "valuation": 0.25,
        "risk": 0.15,
    }

    score = (
        trend_score * weights["trend"]
        + momentum_score * weights["momentum"]
        + quality_score * weights["quality"]
        + valuation_score * weights["valuation"]
        + risk_score * weights["risk"]
    )

    return clamp(score)


# ============================================================
# Signal Engine
# ============================================================

def generate_signal(
    composite_score: float,
    trend_score: float,
    risk_score: float
) -> str:

    if risk_score < 25:
        return "HIGH RISK"

    if composite_score >= 80 and trend_score >= 65:
        return "STRONG BUY"

    if composite_score >= 68:
        return "BUY"

    if composite_score >= 55:
        return "WATCH"

    if composite_score >= 42:
        return "HOLD"

    if composite_score >= 30:
        return "REDUCE"

    return "SELL"


def calculate_confidence(
    composite_score: float,
    trend_score: float,
    momentum_score: float,
    risk_score: float
) -> float:

    scores = np.array([
        composite_score,
        trend_score,
        momentum_score,
        risk_score
    ])

    dispersion = float(np.std(scores))

    confidence = 100 - dispersion * 1.5

    return clamp(confidence, 20, 95)


def risk_label(risk_score: float) -> str:

    if risk_score >= 75:
        return "LOW"

    if risk_score >= 55:
        return "MODERATE"

    if risk_score >= 35:
        return "HIGH"

    return "VERY HIGH"


# ============================================================
# Master Analysis Function
# ============================================================

def analyze_stock(
    ticker: str,
    data: pd.DataFrame,
    fundamentals: Optional[Dict[str, Any]] = None,
) -> AnalysisResult:
    """
    Run the complete V13.1 Foundation analysis.
    """

    df = add_indicators(data)

    if df.empty:
        raise ValueError(
            f"No valid market data available for {ticker}"
        )

    price = safe_float(
        df["Close"].iloc[-1]
    )

    trend_score = calculate_trend_score(df)

    momentum_score = calculate_momentum_score(df)

    risk_score = calculate_risk_score(df)

    valuation_score = calculate_valuation_score(
        fundamentals
    )

    quality_score = calculate_quality_score(
        fundamentals
    )

    composite_score = calculate_composite_score(
        trend_score,
        momentum_score,
        quality_score,
        valuation_score,
        risk_score,
    )

    signal = generate_signal(
        composite_score,
        trend_score,
        risk_score
    )

    confidence = calculate_confidence(
        composite_score,
        trend_score,
        momentum_score,
        risk_score,
    )

    return AnalysisResult(
        ticker=ticker.upper(),
        price=price,
        trend_score=round(trend_score, 2),
        momentum_score=round(momentum_score, 2),
        quality_score=round(quality_score, 2),
        valuation_score=round(valuation_score, 2),
        risk_score=round(risk_score, 2),
        composite_score=round(composite_score, 2),
        signal=signal,
        confidence=round(confidence, 2),
        risk_level=risk_label(risk_score),
    )
