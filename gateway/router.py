"""
================================================================================
   HALCYON CREDIT — LLM Gateway (OpenRouter via direct HTTP)
   Stage 3 | Author: Himkar
   Central LLM router used by ALL agents that need an LLM call.
   Agents import cheap_llm_call() or strong_llm_call() from here.

   NOTE: Uses direct requests to OpenRouter API (bypasses LiteLLM 1.40.20
   bug where 'exception_provider referenced before assignment' is raised
   on any API error). Functionally identical — full OpenAI-compatible endpoint.

   Models (OpenRouter IDs):
     CHEAP_MODEL  : openai/gpt-4.1-mini  (fast, cheap, good for policy/judge)
     STRONG_MODEL : openai/gpt-4.1       (strong, used for synthesizer)
================================================================================
"""
from __future__ import annotations
import os, time, json, requests
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — pulled from .env
# ─────────────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1/chat/completions"
CHEAP_MODEL        = os.getenv("CHEAP_MODEL",  "openai/gpt-4.1-mini")
STRONG_MODEL       = os.getenv("STRONG_MODEL", "openai/gpt-4.1")
COST_CEILING       = float(os.getenv("COST_CEILING_USD", "0.10"))

_HEADERS = {
    "Authorization":  f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type":   "application/json",
    "HTTP-Referer":   "https://github.com/harshit234/Credit-Decision-Intelligence",
    "X-Title":        "Halcyon Credit Agentic Underwriting Copilot",
}


# ─────────────────────────────────────────────────────────────────────────────
# CORE CALL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
def _llm_call(
    model:         str,
    system_prompt: str,
    user_prompt:   str,
    max_tokens:    int   = 1024,
    temperature:   float = 0.1,
    agent_name:    str   = "unknown",
    application_id: str  = "unknown",
) -> tuple[str, float, float]:
    """
    Internal LLM call via direct OpenRouter HTTP (no LiteLLM).

    Returns:
        (response_text, cost_usd, latency_ms)
    Raises:
        RuntimeError on API error or non-200 status
    """
    payload = {
        "model":       model,
        "messages":    [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens":  max_tokens,
        "temperature": temperature,
        # Ask OpenRouter to include real cost accounting in the usage block
        "usage":       {"include": True},
    }

    # Refresh auth header in case key was loaded after module import
    headers = {**_HEADERS, "Authorization": f"Bearer {OPENROUTER_API_KEY}"}

    t0 = time.time()
    try:
        resp       = requests.post(OPENROUTER_BASE, headers=headers,
                                   json=payload, timeout=60)
        latency_ms = (time.time() - t0) * 1000

        if resp.status_code != 200:
            err_msg = resp.json().get("error", {}).get("message", resp.text[:200])
            raise RuntimeError(f"OpenRouter API error {resp.status_code}: {err_msg}")

        data     = resp.json()
        text     = data["choices"][0]["message"]["content"].strip()
        cost_usd = float(data.get("usage", {}).get("cost", 0.0) or 0.0)
        tokens   = data.get("usage", {}).get("total_tokens", 0)

        print(f"  [LLM:{agent_name}] model={model} tokens={tokens} "
              f"cost=${cost_usd:.5f} latency={latency_ms:.0f}ms")

        return text, cost_usd, latency_ms

    except requests.exceptions.RequestException as e:
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
    Call the cheap model path (gpt-4.1-mini via OpenRouter).
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
    Call the strong model path (gpt-4.1 via OpenRouter).
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
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"[{agent_name}] LLM returned invalid JSON: {e}\nRaw: {text[:300]}")
