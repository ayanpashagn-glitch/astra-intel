"""Optional LLM layer.

Three modes, chosen from environment variables (see .env.example):
  * anthropic  – ANTHROPIC_API_KEY set
  * openai     – LLM_API_KEY (+ LLM_BASE_URL, LLM_MODEL) set; works with any
                 OpenAI-compatible server, including a local Ollama
  * local      – nothing set: the app still works using extractive answers

The LLM only ever *phrases* an answer from retrieved passages; retrieval,
page citations and the "not in document" decision do not depend on it.
"""
from __future__ import annotations

import os

import requests

TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "60"))


class LLMError(Exception):
    pass


MODES = ("local", "anthropic", "openai")


def providers() -> dict:
    """Which modes are usable right now, based on environment variables only."""
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    return {
        "local": {"configured": True, "model": "extractive (BM25)", "hint": "Always available. Quotes sentences from the document."},
        "anthropic": {"configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
                      "model": os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001"),
                      "hint": "Set ANTHROPIC_API_KEY in .env"},
        "openai": {"configured": bool(os.environ.get("LLM_API_KEY") or os.environ.get("LLM_BASE_URL")),
                   "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"), "endpoint": base,
                   "hint": "Set LLM_API_KEY (and LLM_BASE_URL / LLM_MODEL for non-OpenAI or local servers)"},
    }


def resolve(mode: str | None = None) -> str:
    """'auto'/None -> best configured mode. Explicit mode must be configured."""
    if mode in (None, "", "auto"):
        p = providers()
        return "anthropic" if p["anthropic"]["configured"] else "openai" if p["openai"]["configured"] else "local"
    if mode not in MODES:
        raise LLMError(f"Unknown mode '{mode}'.")
    if not providers()[mode]["configured"]:
        raise NotConfigured(mode)
    return mode


class NotConfigured(LLMError):
    def __init__(self, mode):
        super().__init__(f"{'Claude (Anthropic)' if mode == 'anthropic' else 'OpenAI-compatible API'} is not configured. "
                         f"{providers()[mode]['hint']}, then restart the server.")
        self.mode = mode


def config(mode: str | None = None) -> dict:
    m = resolve(mode)
    return {"mode": m, "model": providers()[m]["model"]}


def available(mode: str | None = None) -> bool:
    return resolve(mode) != "local"


def complete(system: str, messages: list[dict], max_tokens: int = 700, mode: str | None = None) -> str:
    """messages: [{'role': 'user'|'assistant', 'content': str}]. Temperature 0 for repeatability."""
    cfg = config(mode)
    try:
        if cfg["mode"] == "anthropic":
            r = requests.post(
                os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com") + "/v1/messages",
                headers={
                    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={"model": cfg["model"], "max_tokens": max_tokens, "temperature": 0,
                      "system": system, "messages": messages},
                timeout=TIMEOUT,
            )
            _raise_for(r)
            blocks = r.json().get("content", [])
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        elif cfg["mode"] == "openai":
            base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            headers = {"content-type": "application/json"}
            if os.environ.get("LLM_API_KEY"):
                headers["authorization"] = f"Bearer {os.environ['LLM_API_KEY']}"
            r = requests.post(
                base + "/chat/completions",
                headers=headers,
                json={"model": cfg["model"], "max_tokens": max_tokens, "temperature": 0,
                      "messages": [{"role": "system", "content": system}, *messages]},
                timeout=TIMEOUT,
            )
            _raise_for(r)
            text = (r.json()["choices"][0]["message"]["content"] or "").strip()
        else:
            raise LLMError("No LLM configured.")
    except requests.RequestException as e:
        raise LLMError(f"Could not reach the LLM service ({type(e).__name__}).") from e
    except (KeyError, IndexError, ValueError) as e:
        raise LLMError("The LLM service returned an unexpected response.") from e
    if not text:
        raise LLMError("The LLM returned an empty response.")
    return text


def _raise_for(r: requests.Response):
    if r.status_code in (401, 403):
        raise LLMError("The LLM API key was rejected (check your key).")
    if r.status_code == 429:
        raise LLMError("The LLM API rate limit was hit. Try again shortly.")
    if r.status_code >= 400:
        raise LLMError(f"The LLM service returned HTTP {r.status_code}.")
