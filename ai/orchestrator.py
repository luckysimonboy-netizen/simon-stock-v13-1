"""
Simon Stock V13.2
AI Brain / Research Orchestrator

Architecture

Market Data
     ↓
Quant Engine
     ↓
Fundamental Evidence
     ↓
Research Agents
     ↓
Contradiction Engine
     ↓
Bull / Bear Debate
     ↓
Investment Committee
     ↓
Structured AI Verdict

Design principles:
- Provider agnostic
- Evidence first
- AI reasoning second
- Never invent unavailable data
- Structured output
- Graceful degradation
- Compatible with V13.1
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import requests


# ============================================================
# Configuration
# ============================================================

DEFAULT_TIMEOUT = 45
DEFAULT_RETRIES = 2

MAX_CONTEXT_CHARS = 22000
MAX_PROMPT_CHARS = 30000
MAX_OUTPUT_TOKENS = 5000

SUPPORTED_PROVIDERS = {
    "gemini",
    "openrouter",
    "groq",
}

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


# ============================================================
# Data Models
# ============================================================

@dataclass
class AIResponse:
    success: bool
    provider: str
    model: str
    content: str
    error: Optional[str] = None
    latency_ms: Optional[int] = None
    usage: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchOpinion:
    agent: str
    thesis: str
    positives: List[str]
    negatives: List[str]
    risks: List[str]
    conclusion: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AIVerdict:
    ticker: str
    verdict: str
    confidence: float
    horizon: str
    business_quality: float
    moat: float
    growth: float
    valuation: float
    risk: float
    thesis: str
    bull_case: str
    base_case: str
    bear_case: str
    key_catalyst: str
    key_risk: str
    invalidation: List[str]
    evidence: List[str]
    uncertainty: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Environment
# ============================================================

def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name)

    if value is None:
        return default

    return str(value).strip()


def get_api_key(provider: Optional[str] = None) -> str:
    """
    Get API key for a specific provider.

    If provider is omitted, return the first configured key.
    """

    if provider == "gemini":
        return (
            get_env("GEMINI_API_KEY")
            or get_env("GOOGLE_API_KEY")
        )

    if provider == "openrouter":
        return get_env("OPENROUTER_API_KEY")

    if provider == "groq":
        return get_env("GROQ_API_KEY")

    candidates = [
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
    ]

    for name in candidates:
        value = get_env(name)

        if value:
            return value

    return ""


# ============================================================
# Provider Detection
# ============================================================

def detect_provider() -> str:
    """
    Detect the best configured provider.

    Priority:
        Gemini
        OpenRouter
        Groq
    """

    if get_api_key("gemini"):
        return "gemini"

    if get_api_key("openrouter"):
        return "openrouter"

    if get_api_key("groq"):
        return "groq"

    return "none"


def get_model(provider: Optional[str] = None) -> str:
    provider = provider or detect_provider()

    if provider == "gemini":
        return get_env(
            "GEMINI_MODEL",
            DEFAULT_GEMINI_MODEL,
        )

    if provider == "openrouter":
        return get_env(
            "OPENROUTER_MODEL",
            DEFAULT_OPENROUTER_MODEL,
        )

    if provider == "groq":
        return get_env(
            "GROQ_MODEL",
            DEFAULT_GROQ_MODEL,
        )

    return ""


# ============================================================
# Context Utilities
# ============================================================

def _safe_json_value(value: Any) -> Any:
    """
    Convert common Python / pandas values into
    JSON-safe representations.
    """

    if value is None:
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    if isinstance(value, dict):
        return {
            str(k): _safe_json_value(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _safe_json_value(v)
            for v in value
        ]

    return value


def sanitize_context(
    context: Dict[str, Any],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Convert research data into compact JSON.

    Historical DataFrames are excluded because they can
    consume enormous amounts of context.
    """

    if not isinstance(context, dict):
        context = {
            "data": context
        }

    clean: Dict[str, Any] = {}

    for key, value in context.items():

        # Never dump raw history into AI context.
        if str(key).lower() in {
            "history",
            "ohlcv",
            "price_history",
        }:
            continue

        # DataFrame-like objects
        if hasattr(value, "to_dict"):
            try:
                value = value.to_dict()
            except Exception:
                value = str(value)

        clean[key] = _safe_json_value(value)

    try:

        text = json.dumps(
            clean,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )

    except Exception:

        text = str(clean)

    if len(text) > max_chars:

        text = (
            text[:max_chars]
            + "\n...[context truncated]"
        )

    return text


def compact_prompt(prompt: str) -> str:
    """
    Prevent accidental oversized prompts.
    """

    if len(prompt) <= MAX_PROMPT_CHARS:
        return prompt

    return (
        prompt[:MAX_PROMPT_CHARS]
        + "\n...[prompt truncated]"
    )


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are Simon Stock AI, an advanced US equity research engine.

Your job is to reason about publicly traded companies using
available evidence.

CORE RULES

1. Never invent unavailable data.
2. Never fabricate earnings, prices, analyst targets,
   news, macro events, or financial ratios.
3. Clearly distinguish facts from assumptions.
4. Explicitly identify uncertainty.
5. Never guarantee investment returns.
6. Separate business quality from stock-price momentum.
7. Treat valuation as a range rather than a magic number.
8. Consider opportunity cost.
9. Always analyze both upside and downside.
10. When evidence conflicts, explain the conflict.
11. If a requested metric is unavailable, say so.
12. Do not hide important risks merely because the thesis is bullish.
13. Do not become bullish merely because momentum is strong.
14. Do not become bearish merely because valuation is high.
15. Quantitative scores are evidence, not truth.

ANALYTICAL FRAMEWORKS

VALUE
- free cash flow
- earnings durability
- balance sheet
- capital allocation
- valuation
- margin of safety

BUSINESS
- business model
- pricing power
- customer switching costs
- network effects
- competitive moat
- management
- shareholder alignment
- opportunity cost

FIRST PRINCIPLES
- underlying economics
- technology
- cost structure
- scalability
- market size
- disruption
- execution constraints
- long-term growth ceiling

EVENT / MARKET
- macro
- rates
- regulation
- tariffs
- geopolitics
- catalysts
- market sentiment
- positioning
- event risk

IMPORTANT

You are an analytical system.

Do not impersonate Warren Buffett,
Charlie Munger, Peter Lynch,
Ray Dalio, or any other real investor.

Use investment frameworks as methodologies,
not as personas.
"""


# ============================================================
# JSON Extraction
# ============================================================

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from a model response.

    Handles:
    - pure JSON
    - markdown JSON blocks
    - surrounding prose
    """

    if not text:
        return None

    cleaned = text.strip()

    # Remove markdown fences.
    cleaned = re.sub(
        r"^```(?:json)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"```$",
        "",
        cleaned,
    )

    cleaned = cleaned.strip()

    # Direct JSON.
    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        pass

    # Search first JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start >= 0 and end > start:

        candidate = cleaned[start:end + 1]

        try:
            parsed = json.loads(candidate)

            if isinstance(parsed, dict):
                return parsed

        except Exception:
            pass

    return None


# ============================================================
# Provider: Gemini
# ============================================================

def call_gemini(
    prompt: str,
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> AIResponse:

    api_key = get_api_key("gemini")

    model = model or get_model("gemini")

    if not api_key:

        return AIResponse(
            success=False,
            provider="gemini",
            model=model,
            content="",
            error="Gemini API key is not configured.",
        )

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
    )

    payload = {
        "systemInstruction": {
            "parts": [
                {
                    "text": SYSTEM_PROMPT
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": compact_prompt(prompt)
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.20,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
        },
    }

    started = time.perf_counter()

    try:

        response = requests.post(
            url,
            params={"key": api_key},
            json=payload,
            timeout=timeout,
        )

        latency = int(
            (time.perf_counter() - started) * 1000
        )

        if response.status_code != 200:

            return AIResponse(
                success=False,
                provider="gemini",
                model=model,
                content="",
                error=(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:800]}"
                ),
                latency_ms=latency,
            )

        data = response.json()

        candidates = data.get(
            "candidates",
            [],
        )

        if not candidates:

            return AIResponse(
                success=False,
                provider="gemini",
                model=model,
                content="",
                error="Gemini returned no candidates.",
                latency_ms=latency,
            )

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        content = "\n".join(
            part.get("text", "")
            for part in parts
            if part.get("text")
        ).strip()

        if not content:

            return AIResponse(
                success=False,
                provider="gemini",
                model=model,
                content="",
                error="Gemini returned empty content.",
                latency_ms=latency,
            )

        usage = data.get(
            "usageMetadata"
        )

        return AIResponse(
            success=True,
            provider="gemini",
            model=model,
            content=content,
            latency_ms=latency,
            usage=usage,
        )

    except requests.RequestException as exc:

        return AIResponse(
            success=False,
            provider="gemini",
            model=model,
            content="",
            error=str(exc),
        )


# ============================================================
# Provider: OpenRouter
# ============================================================

def call_openrouter(
    prompt: str,
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> AIResponse:

    api_key = get_api_key("openrouter")

    model = model or get_model("openrouter")

    if not api_key:

        return AIResponse(
            success=False,
            provider="openrouter",
            model=model,
            content="",
            error="OpenRouter API key is not configured.",
        )

    url = (
        "https://openrouter.ai/api/v1/"
        "chat/completions"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "Simon Stock V13.2",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": compact_prompt(prompt),
            },
        ],
        "temperature": 0.20,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }

    started = time.perf_counter()

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )

        latency = int(
            (time.perf_counter() - started) * 1000
        )

        if response.status_code != 200:

            return AIResponse(
                success=False,
                provider="openrouter",
                model=model,
                content="",
                error=(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:800]}"
                ),
                latency_ms=latency,
            )

        data = response.json()

        choices = data.get(
            "choices",
            [],
        )

        if not choices:

            return AIResponse(
                success=False,
                provider="openrouter",
                model=model,
                content="",
                error="OpenRouter returned no choices.",
                latency_ms=latency,
            )

        content = (
            choices[0]
            .get("message", {})
            .get("content", "")
        )

        if not content:

            return AIResponse(
                success=False,
                provider="openrouter",
                model=model,
                content="",
                error="OpenRouter returned empty content.",
                latency_ms=latency,
            )

        return AIResponse(
            success=True,
            provider="openrouter",
            model=model,
            content=str(content),
            latency_ms=latency,
            usage=data.get("usage"),
        )

    except requests.RequestException as exc:

        return AIResponse(
            success=False,
            provider="openrouter",
            model=model,
            content="",
            error=str(exc),
        )


# ============================================================
# Provider: Groq
# ============================================================

def call_groq(
    prompt: str,
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> AIResponse:

    api_key = get_api_key("groq")

    model = model or get_model("groq")

    if not api_key:

        return AIResponse(
            success=False,
            provider="groq",
            model=model,
            content="",
            error="Groq API key is not configured.",
        )

    url = (
        "https://api.groq.com/openai/v1/"
        "chat/completions"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": compact_prompt(prompt),
            },
        ],
        "temperature": 0.20,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }

    started = time.perf_counter()

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )

        latency = int(
            (time.perf_counter() - started) * 1000
        )

        if response.status_code != 200:

            return AIResponse(
                success=False,
                provider="groq",
                model=model,
                content="",
                error=(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:800]}"
                ),
                latency_ms=latency,
            )

        data = response.json()

        choices = data.get(
            "choices",
            [],
        )

        if not choices:

            return AIResponse(
                success=False,
                provider="groq",
                model=model,
                content="",
                error="Groq returned no choices.",
                latency_ms=latency,
            )

        content = (
            choices[0]
            .get("message", {})
            .get("content", "")
        )

        if not content:

            return AIResponse(
                success=False,
                provider="groq",
                model=model,
                content="",
                error="Groq returned empty content.",
                latency_ms=latency,
            )

        return AIResponse(
            success=True,
            provider="groq",
            model=model,
            content=str(content),
            latency_ms=latency,
            usage=data.get("usage"),
        )

    except requests.RequestException as exc:

        return AIResponse(
            success=False,
            provider="groq",
            model=model,
            content="",
            error=str(exc),
        )


# ============================================================
# Generic AI Call
# ============================================================

def call_ai(
    prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    retries: int = DEFAULT_RETRIES,
) -> AIResponse:
    """
    Unified AI entry point.
    """

    provider = (
        provider
        or detect_provider()
    )

    if provider == "none":

        return AIResponse(
            success=False,
            provider="none",
            model="",
            content="",
            error=(
                "No AI provider configured. "
                "Please configure an API key."
            ),
        )

    if provider not in SUPPORTED_PROVIDERS:

        return AIResponse(
            success=False,
            provider=provider,
            model=model or "",
            content="",
            error=(
                f"Unsupported AI provider: {provider}"
            ),
        )

    last_response: Optional[AIResponse] = None

    for attempt in range(
        max(0, retries) + 1
    ):

        if provider == "gemini":

            response = call_gemini(
                prompt,
                model=model,
            )

        elif provider == "openrouter":

            response = call_openrouter(
                prompt,
                model=model,
            )

        else:

            response = call_groq(
                prompt,
                model=model,
            )

        last_response = response

        if response.success:
            return response

        if attempt < retries:

            time.sleep(
                1.5 * (attempt + 1)
            )

    return last_response or AIResponse(
        success=False,
        provider=provider,
        model=model or get_model(provider),
        content="",
        error="Unknown AI error.",
    )


# ============================================================
# Research Agents
# ============================================================

AGENT_PROMPTS = {

    "value": """
You are the VALUE RESEARCH AGENT.

Analyze the company from a long-term value perspective.

Evaluate:

- earnings durability
- free cash flow
- balance sheet
- capital allocation
- valuation
- margin of safety
- opportunity cost
- 3-5 year quality

Do not assume that a high-quality business is automatically
a good stock purchase at any price.
""",

    "business": """
You are the BUSINESS QUALITY AGENT.

Analyze the underlying business.

Evaluate:

- business model
- pricing power
- customer switching costs
- network effects
- competitive moat
- management
- shareholder alignment
- capital allocation
- competitive threats

Focus on whether the business can remain economically strong.
""",

    "first_principles": """
You are the FIRST-PRINCIPLES AGENT.

Break the business into fundamental economic drivers.

Evaluate:

- market size
- unit economics
- cost structure
- technology
- scalability
- innovation
- disruption
- execution constraints
- growth ceiling
- long-term industry structure

Do not rely solely on historical growth.
Ask what must be true for future growth to occur.
""",

    "event": """
You are the EVENT / MARKET AGENT.

Evaluate short- and medium-term drivers.

Consider:

- interest rates
- macro environment
- regulation
- tariffs
- geopolitics
- earnings catalysts
- product launches
- market sentiment
- positioning
- event risks

Only discuss events that are present in the supplied evidence.
If current news is unavailable, explicitly say so.
""",
}


# ============================================================
# Research Agent Runner
# ============================================================

def run_research_agent(
    agent_name: str,
    ticker: str,
    context: Dict[str, Any],
) -> ResearchOpinion:

    if agent_name not in AGENT_PROMPTS:

        raise ValueError(
            f"Unknown research agent: {agent_name}"
        )

    compact_context = sanitize_context(
        context
    )

    prompt = f"""
Ticker:
{ticker}

AVAILABLE EVIDENCE:
{compact_context}

ANALYTICAL ROLE:
{AGENT_PROMPTS[agent_name]}

Return a rigorous research opinion.

Required structure:

THESIS:
...

POSITIVES:
- ...
- ...
- ...

NEGATIVES:
- ...
- ...
- ...

RISKS:
- ...
- ...
- ...

CONCLUSION:
...

Never invent missing evidence.
"""

    response = call_ai(prompt)

    if not response.success:

        return ResearchOpinion(
            agent=agent_name,
            thesis="AI analysis unavailable.",
            positives=[],
            negatives=[],
            risks=[
                response.error
                or "Unknown AI error."
            ],
            conclusion="No AI conclusion available.",
        )

    return ResearchOpinion(
        agent=agent_name,
        thesis=response.content,
        positives=[],
        negatives=[],
        risks=[],
        conclusion=response.content,
    )


# ============================================================
# Contradiction Engine
# ============================================================

def run_contradiction_check(
    ticker: str,
    context: Dict[str, Any],
    opinions: List[ResearchOpinion],
) -> AIResponse:

    compact_context = sanitize_context(
        context
    )

    research = "\n\n".join(
        (
            f"=== {op.agent.upper()} ===\n"
            f"{op.conclusion}"
        )
        for op in opinions
    )

    prompt = f"""
Ticker:
{ticker}

AVAILABLE EVIDENCE:
{compact_context}

RESEARCH AGENTS:
{research}

You are the CONTRADICTION DETECTION ENGINE.

Look for conflicts such as:

- strong business + expensive valuation
- strong momentum + weak fundamentals
- high growth + deteriorating margins
- high quality + increasing risk
- bullish narrative + weak evidence
- bearish narrative + improving fundamentals

Return:

1. Major contradictions
2. Which evidence is stronger
3. What cannot currently be determined
4. What additional data would resolve the conflict
5. Whether the contradiction materially changes the investment thesis

Be skeptical.
Do not force agreement.
"""

    return call_ai(prompt)


# ============================================================
# Bull / Bear Debate
# ============================================================

def run_bull_bear_debate(
    ticker: str,
    context: Dict[str, Any],
    opinions: List[ResearchOpinion],
) -> AIResponse:

    compact_context = sanitize_context(
        context
    )

    research = "\n\n".join(
        (
            f"=== {op.agent.upper()} ===\n"
            f"{op.conclusion}"
        )
        for op in opinions
    )

    prompt = f"""
Ticker:
{ticker}

AVAILABLE EVIDENCE:
{compact_context}

RESEARCH:
{research}

Run an adversarial investment debate.

BULL CASE
- strongest reason to own
- growth drivers
- competitive advantages
- valuation upside
- catalysts

BEAR CASE
- strongest reason not to own
- valuation risk
- competitive threats
- macro risk
- execution risk

Then answer:

1. What is the bull case underestimating?
2. What is the bear case underestimating?
3. What evidence would invalidate the bull case?
4. What evidence would invalidate the bear case?
5. What is the single most important unknown?

Do not average the opinions.
Challenge them.
"""

    return call_ai(prompt)


# ============================================================
# Investment Committee
# ============================================================

COMMITTEE_SCHEMA = {
    "verdict": "STRONG BUY | BUY | WATCH | HOLD | REDUCE | SELL",
    "confidence": 0,
    "horizon": "string",
    "business_quality": 0,
    "moat": 0,
    "growth": 0,
    "valuation": 0,
    "risk": 0,
    "thesis": "string",
    "bull_case": "string",
    "base_case": "string",
    "bear_case": "string",
    "key_catalyst": "string",
    "key_risk": "string",
    "invalidation": [],
    "evidence": [],
    "uncertainty": [],
}


def _number(
    value: Any,
    default: float = 50.0,
) -> float:

    try:
        value = float(value)
    except Exception:
        value = default

    return max(
        0.0,
        min(100.0, value),
    )


def _string(
    value: Any,
    default: str = "",
) -> str:

    if value is None:
        return default

    return str(value).strip()


def _list(
    value: Any,
) -> List[str]:

    if value is None:
        return []

    if isinstance(value, list):

        return [
            str(item)
            for item in value
            if item is not None
        ]

    return [str(value)]


def parse_ai_verdict(
    ticker: str,
    content: str,
) -> AIVerdict:

    data = extract_json(content)

    if not data:
        return AIVerdict(
            ticker=ticker.upper(),
            verdict="WATCH",
            confidence=35,
            horizon="Unknown",
            business_quality=50,
            moat=50,
            growth=50,
            valuation=50,
            risk=50,
            thesis=content[:3000],
            bull_case="Unavailable.",
            base_case="Unavailable.",
            bear_case="Unavailable.",
            key_catalyst="Unavailable.",
            key_risk="AI output could not be structured.",
            invalidation=[],
            evidence=[],
            uncertainty=[
                "AI response was not returned as valid JSON."
            ],
        )

    verdict = _string(
        data.get("verdict"),
        "WATCH",
    ).upper()

    allowed = {
        "STRONG BUY",
        "BUY",
        "WATCH",
        "HOLD",
        "REDUCE",
        "SELL",
    }

    if verdict not in allowed:
        verdict = "WATCH"

    return AIVerdict(
        ticker=ticker.upper(),
        verdict=verdict,
        confidence=_number(
            data.get("confidence"),
            50,
        ),
        horizon=_string(
            data.get("horizon"),
            "Unknown",
        ),
        business_quality=_number(
            data.get("business_quality"),
        ),
        moat=_number(
            data.get("moat"),
        ),
        growth=_number(
            data.get("growth"),
        ),
        valuation=_number(
            data.get("valuation"),
        ),
        risk=_number(
            data.get("risk"),
        ),
        thesis=_string(
            data.get("thesis"),
            "No thesis provided.",
        ),
        bull_case=_string(
            data.get("bull_case"),
            "Unavailable.",
        ),
        base_case=_string(
            data.get("base_case"),
            "Unavailable.",
        ),
        bear_case=_string(
            data.get("bear_case"),
            "Unavailable.",
        ),
        key_catalyst=_string(
            data.get("key_catalyst"),
            "Unknown.",
        ),
        key_risk=_string(
            data.get("key_risk"),
            "Unknown.",
        ),
        invalidation=_list(
            data.get("invalidation")
        ),
        evidence=_list(
            data.get("evidence")
        ),
        uncertainty=_list(
            data.get("uncertainty")
        ),
    )


def run_investment_committee(
    ticker: str,
    context: Dict[str, Any],
    opinions: List[ResearchOpinion],
    debate: Optional[AIResponse] = None,
    contradictions: Optional[AIResponse] = None,
) -> AIResponse:

    compact_context = sanitize_context(
        context
    )

    research = "\n\n".join(
        (
            f"=== {op.agent.upper()} ===\n"
            f"{op.conclusion}"
        )
        for op in opinions
    )

    debate_text = (
        debate.content
        if debate and debate.success
        else "Bull/Bear debate unavailable."
    )

    contradiction_text = (
        contradictions.content
        if contradictions and contradictions.success
        else "Contradiction analysis unavailable."
    )

    schema_text = json.dumps(
        COMMITTEE_SCHEMA,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
You are the CHIEF INVESTMENT ANALYST of Simon Stock.

Ticker:
{ticker}

AVAILABLE EVIDENCE:
{compact_context}

RESEARCH:
{research}

BULL / BEAR DEBATE:
{debate_text}

CONTRADICTION ANALYSIS:
{contradiction_text}

Your task is to make the final investment-research judgment.

IMPORTANT:

The quantitative and fundamental evidence should constrain
your conclusion.

Do not produce a bullish verdict merely because the company
is famous or has strong historical performance.

Do not produce a bearish verdict merely because valuation
is elevated.

If evidence is insufficient, choose WATCH.

Return ONLY valid JSON.

Use exactly this structure:

{schema_text}

SCORING

business_quality:
0-100

moat:
0-100

growth:
0-100

valuation:
0-100

risk:
0-100

confidence:
0-100

VERDICT OPTIONS:

STRONG BUY
BUY
WATCH
HOLD
REDUCE
SELL

The "risk" score means:

100 = low risk
0 = extremely high risk

The valuation score means:

100 = very attractive valuation
0 = extremely unattractive valuation

The final thesis must explicitly mention
the most important trade-off.

Evidence must only contain evidence available
in the supplied context or derived transparently
from it.

Uncertainty must list important unknowns.

This is research, not guaranteed or personalized
financial advice.
"""

    return call_ai(
        compact_prompt(prompt)
    )


# ============================================================
# Full AI Pipeline
# ============================================================

def run_full_ai_research(
    ticker: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute full V13.2 AI research.

    Pipeline:

        Value
        Business
        First Principles
        Event
             ↓
        Contradiction
             ↓
        Bull / Bear
             ↓
        Investment Committee
    """

    symbol = str(
        ticker
    ).strip().upper()

    opinions: List[ResearchOpinion] = []

    agents = [
        "value",
        "business",
        "first_principles",
        "event",
    ]

    for agent_name in agents:

        opinion = run_research_agent(
            agent_name,
            symbol,
            context,
        )

        opinions.append(
            opinion
        )

    contradictions = run_contradiction_check(
        symbol,
        context,
        opinions,
    )

    debate = run_bull_bear_debate(
        symbol,
        context,
        opinions,
    )

    committee = run_investment_committee(
        symbol,
        context,
        opinions,
        debate,
        contradictions,
    )

    verdict = parse_ai_verdict(
        symbol,
        committee.content
        if committee.success
        else "",
    )

    return {
        "ticker": symbol,
        "provider": detect_provider(),
        "model": get_model(),
        "agents": [
            opinion.to_dict()
            for opinion in opinions
        ],
        "contradictions": contradictions.to_dict(),
        "debate": debate.to_dict(),
        "committee": committee.to_dict(),
        "verdict": verdict.to_dict(),
    }


# ============================================================
# Lightweight AI Analysis
# ============================================================

def run_quick_ai_analysis(
    ticker: str,
    context: Dict[str, Any],
) -> AIResponse:
    """
    Lightweight single-call analysis.

    Useful for:
    - mobile UI
    - quick stock search
    - watchlist scanning
    - daily market dashboard
    """

    compact_context = sanitize_context(
        context,
        max_chars=12000,
    )

    prompt = f"""
Analyze {ticker} using the supplied evidence.

Give:

1. One-sentence verdict
2. Three strongest positives
3. Three biggest risks
4. Valuation assessment
5. Short-term momentum assessment
6. Long-term business assessment
7. One thing that could change the thesis

Do not invent data.

Evidence:
{compact_context}
"""

    return call_ai(
        prompt
    )


# ============================================================
# Provider Fallback
# ============================================================

def call_ai_with_fallback(
    prompt: str,
    preferred_provider: Optional[str] = None,
    retries: int = 1,
) -> AIResponse:
    """
    Try preferred provider first.

    If it fails, automatically try the remaining
    configured providers.
    """

    providers: List[str] = []

    if preferred_provider in SUPPORTED_PROVIDERS:
        providers.append(
            preferred_provider
        )

    detected = detect_provider()

    if detected in SUPPORTED_PROVIDERS:
        providers.append(
            detected
        )

    for provider in [
        "gemini",
        "openrouter",
        "groq",
    ]:
        if provider not in providers:
            providers.append(provider)

    last_response: Optional[AIResponse] = None

    for provider in providers:

        if not get_api_key(provider):
            continue

        response = call_ai(
            prompt,
            provider=provider,
            retries=retries,
        )

        last_response = response

        if response.success:
            return response

    return last_response or AIResponse(
        success=False,
        provider="none",
        model="",
        content="",
        error="No configured AI provider succeeded.",
    )


# ============================================================
# AI Health Check
# ============================================================

def ai_health_check() -> Dict[str, Any]:
    """
    Configuration health check.

    Does not consume API credits.
    """

    configured = []

    for provider in [
        "gemini",
        "openrouter",
        "groq",
    ]:

        if get_api_key(provider):

            configured.append({
                "provider": provider,
                "model": get_model(provider),
            })

    primary = detect_provider()

    return {
        "configured": bool(configured),
        "provider": primary,
        "model": get_model(primary),
        "providers": configured,
        "message": (
            "AI provider configured."
            if configured
            else
            "No AI provider configured."
        ),
    }


# ============================================================
# Backward Compatibility
# ============================================================

def get_ai_status() -> Dict[str, Any]:
    """
    Alias kept for future app versions.
    """

    return ai_health_check()


def test_ai_connection() -> AIResponse:
    """
    Optional live connection test.

    This DOES consume a small amount of API usage.
    """

    prompt = """
Return exactly:

Simon Stock AI connection OK.
"""

    return call_ai(
        prompt,
        retries=1,
    )
