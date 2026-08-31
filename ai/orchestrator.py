"""
Simon Stock V13.1 Foundation
AI Orchestrator

AI architecture:

Market / Fundamental Data
            ↓
      Research Agents
            ↓
 ┌──────────┼──────────┐
 ↓          ↓          ↓
Value    Business    Growth
Agent    Agent       Agent
            ↓
       Event Agent
            ↓
      Bull / Bear Debate
            ↓
   Investment Committee
            ↓
      Final AI Report

The AI layer is deliberately provider-agnostic.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import requests


# ============================================================
# Configuration
# ============================================================

DEFAULT_TIMEOUT = 45
DEFAULT_RETRIES = 2

MAX_CONTEXT_CHARS = 18000
MAX_OUTPUT_TOKENS = 4000


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


# ============================================================
# Environment Helpers
# ============================================================

def get_env(name: str, default: str = "") -> str:
    """
    Safely retrieve an environment variable.
    """

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip()


def get_api_key() -> str:
    """
    Read the primary AI key.

    Supported environment names:

        GEMINI_API_KEY
        GOOGLE_API_KEY
        OPENROUTER_API_KEY
        GROQ_API_KEY

    Priority:

        GEMINI_API_KEY
        GOOGLE_API_KEY
        OPENROUTER_API_KEY
        GROQ_API_KEY
    """

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
    Detect the configured AI provider.

    Priority:
        Gemini
        OpenRouter
        Groq
    """

    if get_env("GEMINI_API_KEY"):
        return "gemini"

    if get_env("GOOGLE_API_KEY"):
        return "gemini"

    if get_env("OPENROUTER_API_KEY"):
        return "openrouter"

    if get_env("GROQ_API_KEY"):
        return "groq"

    return "none"


def get_model(provider: Optional[str] = None) -> str:
    """
    Get configured model.

    Environment overrides:

        GEMINI_MODEL
        OPENROUTER_MODEL
        GROQ_MODEL
    """

    provider = provider or detect_provider()

    if provider == "gemini":
        return get_env(
            "GEMINI_MODEL",
            "gemini-2.5-flash"
        )

    if provider == "openrouter":
        return get_env(
            "OPENROUTER_MODEL",
            "google/gemini-2.5-flash"
        )

    if provider == "groq":
        return get_env(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile"
        )

    return ""


# ============================================================
# Context Sanitization
# ============================================================

def sanitize_context(
    context: Dict[str, Any],
    max_chars: int = MAX_CONTEXT_CHARS
) -> str:
    """
    Convert stock data into compact JSON.

    Large historical datasets are intentionally not
    dumped into the model context.
    """

    clean = {}

    for key, value in context.items():

        if key == "history":
            continue

        if hasattr(value, "to_dict"):
            try:
                value = value.to_dict()
            except Exception:
                value = str(value)

        clean[key] = value

    try:

        text = json.dumps(
            clean,
            ensure_ascii=False,
            default=str,
        )

    except Exception:

        text = str(clean)

    if len(text) > max_chars:

        text = (
            text[:max_chars]
            + "\n...[context truncated]"
        )

    return text


# ============================================================
# Core Prompt
# ============================================================

SYSTEM_PROMPT = """
You are Simon Stock AI, an advanced US equity research assistant.

Your job is to analyze companies using evidence rather than hype.

Important rules:

1. Never pretend that unavailable data exists.
2. Separate facts from assumptions.
3. Clearly identify uncertainty.
4. Never guarantee returns.
5. Distinguish investment quality from short-term price momentum.
6. Explain the reasoning behind every important conclusion.
7. When evidence conflicts, explicitly describe the conflict.
8. Consider both bullish and bearish scenarios.
9. Treat valuation as a range, not a magic number.
10. Risk management is part of the conclusion.

You may use four analytical lenses:

VALUE:
- moat
- cash flow
- capital allocation
- valuation
- margin of safety

BUSINESS:
- business model
- pricing power
- management
- corporate culture
- shareholder alignment
- opportunity cost

FIRST PRINCIPLES:
- underlying economics
- technological disruption
- cost structure
- market size
- scalability
- execution constraints

EVENT / MARKET:
- macro environment
- policy
- regulation
- tariffs
- market sentiment
- catalysts
- positioning

Do not impersonate or claim to be any real investor.

Use the frameworks as analytical methodologies only.
"""


# ============================================================
# Provider: Gemini
# ============================================================

def call_gemini(
    prompt: str,
    model: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> AIResponse:

    api_key = (
        get_env("GEMINI_API_KEY")
        or get_env("GOOGLE_API_KEY")
    )

    if not api_key:

        return AIResponse(
            success=False,
            provider="gemini",
            model=model or "",
            content="",
            error="Gemini API key is not configured.",
        )

    model = model or get_model("gemini")

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
    )

    params = {
        "key": api_key
    }

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
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.25,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
        },
    }

    started = time.perf_counter()

    try:

        response = requests.post(
            url,
            params=params,
            json=payload,
            timeout=timeout,
        )

        latency = int(
            (time.perf_counter() - started)
            * 1000
        )

        if response.status_code != 200:

            return AIResponse(
                success=False,
                provider="gemini",
                model=model,
                content="",
                error=(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                ),
                latency_ms=latency,
            )

        data = response.json()

        candidates = data.get(
            "candidates",
            []
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

        return AIResponse(
            success=True,
            provider="gemini",
            model=model,
            content=content,
            latency_ms=latency,
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

    api_key = get_env(
        "OPENROUTER_API_KEY"
    )

    if not api_key:

        return AIResponse(
            success=False,
            provider="openrouter",
            model=model or "",
            content="",
            error="OpenRouter API key is not configured.",
        )

    model = model or get_model("openrouter")

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "Simon Stock V13.1",
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
                "content": prompt,
            },
        ],
        "temperature": 0.25,
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
            (time.perf_counter() - started)
            * 1000
        )

        if response.status_code != 200:

            return AIResponse(
                success=False,
                provider="openrouter",
                model=model,
                content="",
                error=(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                ),
                latency_ms=latency,
            )

        data = response.json()

        choices = data.get(
            "choices",
            []
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

    api_key = get_env(
        "GROQ_API_KEY"
    )

    if not api_key:

        return AIResponse(
            success=False,
            provider="groq",
            model=model or "",
            content="",
            error="Groq API key is not configured.",
        )

    model = model or get_model("groq")

    url = "https://api.groq.com/openai/v1/chat/completions"

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
                "content": prompt,
            },
        ],
        "temperature": 0.25,
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
            (time.perf_counter() - started)
            * 1000
        )

        if response.status_code != 200:

            return AIResponse(
                success=False,
                provider="groq",
                model=model,
                content="",
                error=(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                ),
                latency_ms=latency,
            )

        data = response.json()

        choices = data.get(
            "choices",
            []
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

        return AIResponse(
            success=True,
            provider="groq",
            model=model,
            content=str(content),
            latency_ms=latency,
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

    Provider selection:
        auto → configured provider

    This makes the rest of the application
    independent from any specific AI company.
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

    last_response = None

    for attempt in range(
        retries + 1
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

        elif provider == "groq":

            response = call_groq(
                prompt,
                model=model,
            )

        else:

            return AIResponse(
                success=False,
                provider=provider,
                model=model or "",
                content="",
                error=(
                    f"Unsupported AI provider: "
                    f"{provider}"
                ),
            )

        last_response = response

        if response.success:
            return response

        if attempt < retries:
            time.sleep(
                1.5 * (attempt + 1)
            )

    return last_response


# ============================================================
# Research Agent Prompts
# ============================================================

AGENT_PROMPTS = {

    "value": """
Act as the Value Research Agent.

Analyze the company through a long-term value-investing lens.

Focus on:
- competitive moat
- durability of earnings
- free cash flow
- capital allocation
- balance sheet
- valuation
- margin of safety
- 3-5 year business quality

Return:
1. Thesis
2. Positives
3. Negatives
4. Key risks
5. Conclusion
""",

    "business": """
Act as the Business Quality Agent.

Analyze the company as if the investor is buying
the underlying business rather than simply buying
a ticker symbol.

Focus on:
- business model
- pricing power
- unit economics
- customer switching costs
- management quality
- shareholder alignment
- corporate culture
- capital allocation
- opportunity cost

Return:
1. Thesis
2. Positives
3. Negatives
4. Key risks
5. Conclusion
""",

    "first_principles": """
Act as the First-Principles Growth Agent.

Break the business down into fundamental economic
and technological components.

Focus on:
- raw economic drivers
- technology
- cost curves
- scalability
- market size
- innovation
- disruption potential
- execution constraints
- long-term growth ceiling

Return:
1. Thesis
2. Positives
3. Negatives
4. Key risks
5. Conclusion
""",

    "event": """
Act as the Event and Market Agent.

Analyze short- and medium-term market drivers.

Focus on:
- macroeconomic environment
- interest rates
- policy
- regulation
- tariffs
- geopolitical exposure
- catalysts
- market sentiment
- positioning
- event-driven upside/downside

Return:
1. Thesis
2. Positives
3. Negatives
4. Key risks
5. Conclusion
""",
}


# ============================================================
# Agent Execution
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
Ticker: {ticker}

Research data:
{compact_context}

{AGENT_PROMPTS[agent_name]}

Do not invent missing financial data.

Use explicit uncertainty where necessary.
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

    opinion_text = "\n\n".join(
        (
            f"[{op.agent.upper()}]\n"
            f"{op.conclusion}"
        )
        for op in opinions
    )

    prompt = f"""
Ticker: {ticker}

Company data:
{compact_context}

Research opinions:
{opinion_text}

Now run a structured Bull vs Bear debate.

BULL CASE:
- strongest reasons to own the company
- growth drivers
- valuation upside
- catalysts

BEAR CASE:
- strongest reasons not to own it
- valuation risks
- competitive threats
- macro risks
- execution risks

Then identify:
- what each side is underestimating
- what evidence would invalidate the bull case
- what evidence would invalidate the bear case
- the most important unknown

Do not simply average the opinions.
Challenge them.
"""

    return call_ai(prompt)


# ============================================================
# Investment Committee
# ============================================================

def run_investment_committee(
    ticker: str,
    context: Dict[str, Any],
    opinions: List[ResearchOpinion],
    debate: Optional[AIResponse] = None,
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

    prompt = f"""
You are the Chief Investment Analyst of Simon Stock.

Ticker:
{ticker}

Data:
{compact_context}

Research:
{research}

Bull/Bear debate:
{debate_text}

Produce the final Investment Committee report.

Use this structure:

# Executive Verdict

Give a concise overall assessment.

# Business Quality

Assess the underlying business.

# Competitive Moat

Explain the strength and durability of the moat.

# Growth

Explain the main growth engines and limitations.

# Valuation

Discuss valuation using available evidence.
Do not invent a DCF value if required inputs are missing.

# Risk

List the five most important risks.

# Bull Case

Describe the optimistic scenario.

# Base Case

Describe the most reasonable scenario.

# Bear Case

Describe the downside scenario.

# What Would Change My Mind

Give concrete indicators that should cause
the thesis to be upgraded or downgraded.

# Decision Framework

Choose one:

STRONG BUY
BUY
WATCH
HOLD
REDUCE
SELL

Then provide:

- confidence: 0-100
- time horizon
- key catalyst
- key risk
- valuation zone if defensible

Important:
This is research, not a guarantee or personalized
financial advice.
"""

    return call_ai(prompt)


# ============================================================
# Full AI Research Pipeline
# ============================================================

def run_full_ai_research(
    ticker: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the complete AI research pipeline.

    Pipeline:

        Value
          ↓
        Business
          ↓
        First Principles
          ↓
        Event
          ↓
        Bull/Bear
          ↓
        Investment Committee
    """

    opinions = []

    for agent_name in [
        "value",
        "business",
        "first_principles",
        "event",
    ]:

        opinion = run_research_agent(
            agent_name,
            ticker,
            context,
        )

        opinions.append(
            opinion
        )

    debate = run_bull_bear_debate(
        ticker,
        context,
        opinions,
    )

    committee = run_investment_committee(
        ticker,
        context,
        opinions,
        debate,
    )

    return {
        "ticker": ticker.upper(),
        "provider": detect_provider(),
        "model": get_model(),
        "agents": [
            opinion.to_dict()
            for opinion in opinions
        ],
        "debate": debate.to_dict(),
        "committee": committee.to_dict(),
    }


# ============================================================
# AI Health Check
# ============================================================

def ai_health_check() -> Dict[str, Any]:
    """
    Return AI configuration status.

    Does not consume API credits.
    """

    provider = detect_provider()
    model = get_model(provider)

    return {
        "configured": provider != "none",
        "provider": provider,
        "model": model,
        "message": (
            "AI provider configured."
            if provider != "none"
            else
            "No AI provider configured."
        ),
    }
