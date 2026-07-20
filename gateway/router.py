"""
================================================================================
   HALCYON CREDIT — LiteLLM Gateway (OpenRouter)
   Stage 3 | Author: Himkar
   Central LLM router used by ALL agents that need an LLM call.
   Agents import cheap_llm_call() or strong_llm_call() from here.
   All OpenRouter config lives here — one place to change model/key.
================================================================================
"""
from __future__ import annotations
import os
import time
import json
import litellm
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — pulled from .env
# ─────────────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1"
CHEAP_MODEL        = os.getenv("CHEAP_MODEL",  "google/gemini-2.0-flash-001")
STRONG_MODEL       = os.getenv("STRONG_MODEL", "google/gemini-2.0-flash-001")
COST_CEILING       = float(os.getenv("COST_CEILING_USD", "0.10"))

# LiteLLM OpenRouter setup
litellm.drop_params = True    # silently ignore unsupported params
litellm.set_verbose = False

# Extra headers OpenRouter recommends
_EXTRA_HEADERS = {
    "HTTP-Referer": "https://github.com/harshit234/Credit-Decision-Intelligence",
    "X-Title":      "Halcyon Credit Agentic Underwriting Copilot",
}


# ─────────────────────────────────────────────────────────────────────────────
# CORE CALL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
def _llm_call(
    model:        str,
    system_prompt: str,
    user_prompt:   str,
    max_tokens:    int   = 1024,
    temperature:   float = 0.1,
    agent_name:    str   = "unknown",
    application_id: str  = "unknown",
) -> tuple[str, float, float]:
    """
    Internal LLM call wrapper.

    Returns:
        (response_text, cost_usd, latency_ms)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    t0 = time.time()
    try:
        response = litellm.completion(
            model          = model,
            messages       = messages,
            max_tokens     = max_tokens,
            temperature    = temperature,
            api_base       = OPENROUTER_BASE,
            api_key        = OPENROUTER_API_KEY,
            extra_headers  = _EXTRA_HEADERS,
        )
        latency_ms = (time.time() - t0) * 1000
        text       = response.choices[0].message.content.strip()

        # Cost tracking
        cost_usd = 0.0
        if hasattr(response, "_hidden_params"):
            cost_usd = response._hidden_params.get("response_cost", 0.0) or 0.0

        print(f"  [LLM:{agent_name}] model={model} tokens={response.usage.total_tokens} "
              f"cost=${cost_usd:.5f} latency={latency_ms:.0f}ms")

        return text, cost_usd, latency_ms

    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        raise RuntimeError(f"LLM call failed ({agent_name}): {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — used by agents
# ─────────────────────────────────────────────────────────────────────────────
def cheap_llm_call(
    system_prompt:  str,
    user_prompt:    str,
    max_tokens:     int = 512,
    agent_name:     str = "unknown",
    application_id: str = "unknown",
) -> tuple[str, float, float]:
    """
    Call the cheap model path (Gemini Flash via OpenRouter).
    Used by: PolicyComplianceAgent
    Returns: (response_text, cost_usd, latency_ms)
    """
    return _llm_call(
        model          = CHEAP_MODEL,
        system_prompt  = system_prompt,
        user_prompt    = user_prompt,
        max_tokens     = max_tokens,
        agent_name     = agent_name,
        application_id = application_id,
    )


def strong_llm_call(
    system_prompt:  str,
    user_prompt:    str,
    max_tokens:     int = 2048,
    agent_name:     str = "unknown",
    application_id: str = "unknown",
) -> tuple[str, float, float]:
    """
    Call the strong model path (Gemini Pro via OpenRouter).
    Used by: DecisionSynthesizerAgent, EvaluationAgent
    Returns: (response_text, cost_usd, latency_ms)
    """
    return _llm_call(
        model          = STRONG_MODEL,
        system_prompt  = system_prompt,
        user_prompt    = user_prompt,
        max_tokens     = max_tokens,
        agent_name     = agent_name,
        application_id = application_id,
    )


def parse_json_response(text: str, agent_name: str) -> dict:
    """
    Safely parse a JSON response from the LLM.
    Strips markdown code fences if present.
    """
    # Strip ```json ... ``` fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"[{agent_name}] LLM returned invalid JSON: {e}\nRaw: {text[:300]}")
