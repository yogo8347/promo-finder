"""One tiny interface for every model: `complete(prompt, system) -> str`.

Swap the model with two settings (CLI flags or env vars):  PROMO_LLM_PROVIDER, PROMO_LLM_MODEL.

  none       no model; the rules engine does everything (default, free, deterministic)
  gemini     Google AI Studio free tier       (GEMINI_API_KEY)          default gemma-4-26b-a4b-it (free; Flash models: ~20 req/day)
  groq       Groq free tier, OpenAI-style     (GROQ_API_KEY)            default llama-3.3-70b-versatile
  openrouter OpenRouter (has free models)     (OPENROUTER_API_KEY)
  ollama     local model, no key              (OLLAMA_HOST)             default qwen2.5:3b
  openai     ANY OpenAI-compatible endpoint   (OPENAI_API_KEY, OPENAI_BASE_URL)  <- e.g. Kuku's own gateway
  anthropic  Claude API                       (ANTHROPIC_API_KEY)

Every response is cached on disk by (provider, model, prompt), so re-runs are free and
reproducible, and a rate-limited free tier is never hit twice for the same question.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

DEFAULT_MODELS = {
    "gemini": "gemma-4-26b-a4b-it",
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
    "ollama": "qwen2.5:3b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
}
OPENAI_STYLE = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
}


def _base_url(provider: str) -> str:
    if provider == "ollama":
        return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/v1"
    if provider == "openai":
        return os.environ.get("OPENAI_BASE_URL", OPENAI_STYLE["openai"][0])
    return OPENAI_STYLE[provider][0]


class LLMError(RuntimeError):
    pass


def _log(msg: str) -> None:
    """Progress/retry lines go to stderr, flushed, so a long run is never silent."""
    import sys
    print(msg, file=sys.stderr, flush=True)


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (no extra dependency). Real environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v and k not in os.environ:
            os.environ[k] = v


load_dotenv()


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 cache_dir: str | Path = ".cache/llm", min_interval: float | None = None,
                 temperature: float = 0.2, timeout: int = 90):
        self.provider = (provider or os.environ.get("PROMO_LLM_PROVIDER") or "none").lower()
        self.model = model or os.environ.get("PROMO_LLM_MODEL") or DEFAULT_MODELS.get(self.provider, "")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        # free tiers are rate limited (Gemini flash ~10 req/min); be polite by default
        default_gap = {"gemini": 4.0, "groq": 2.5, "openrouter": 4.0}.get(self.provider, 0.0)
        self.min_interval = default_gap if min_interval is None else min_interval
        self.temperature = temperature
        self.timeout = timeout
        self._last = 0.0
        # if the free tier's daily cap is hit, continue on a cheaper model with its own quota
        # (Gemini free tier, Nov 2026: Gemini Flash models allow only ~20 requests/day each;
        #  Gemma models on the same key allow far more, so Gemma is the free default)
        fb = os.environ.get("PROMO_LLM_FALLBACKS")
        if fb is not None:
            self.fallbacks = [m.strip() for m in fb.split(",") if m.strip()]
        elif self.provider == "gemini" and self.model.startswith("gemma"):
            self.fallbacks = ["gemma-4-26b-a4b-it", "gemma-4-31b-it"]
        elif self.provider == "gemini":
            self.fallbacks = ["gemini-3.8-flash", "gemini-flash-lite-latest", "gemini-3.5-flash-lite"]
        else:
            self.fallbacks = []
        self.fallbacks = [m for m in self.fallbacks if m != self.model]
        self._daily_quota_hit = False
        self.first_model = self.model
        self.events: list[str] = []
        self.calls = 0
        self.cache_hits = 0
        self.tokens_in = 0
        self.tokens_out = 0

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    def describe(self) -> str:
        if not self.enabled:
            return "rules only (no LLM)"
        used = [self.first_model] + [e.split("switched to ")[-1] for e in self.events if "switched to" in e]
        return f"{self.provider}:{'+'.join(dict.fromkeys(used))}"

    # ---------------------------------------------------------------- public ----------
    def complete(self, prompt: str, system: str = "", want_json: bool = True) -> str:
        if not self.enabled:
            raise LLMError("LLM provider is 'none'")
        key = hashlib.sha256(json.dumps([self.provider, self.model, system, prompt]).encode()).hexdigest()[:24]
        f = self.cache / f"{key}.json"
        if f.exists():
            self.cache_hits += 1
            return json.loads(f.read_text(encoding="utf-8"))["text"]
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        t0 = time.time()
        text = self._with_retries(prompt, system, want_json)
        self._last = time.time()
        self.calls += 1
        _log(f"[llm] call {self.calls} on {self.model}: {self._last - t0:.1f}s")
        f.write_text(json.dumps({"provider": self.provider, "model": self.model, "text": text}), encoding="utf-8")
        return text

    def complete_json(self, prompt: str, system: str = "") -> dict | list:
        return parse_json(self.complete(prompt, system, want_json=True))

    # ---------------------------------------------------------------- backends --------
    def _switch(self, why: str) -> bool:
        """Move to the next fallback model (sticky for the rest of the run)."""
        if not self.fallbacks:
            return False
        old, self.model = self.model, self.fallbacks.pop(0)
        self.events.append(f"{why} on {old}; switched to {self.model}")
        _log(f"[llm] {self.events[-1]}")
        return True

    def _with_retries(self, prompt, system, want_json, tries: int = 9) -> str:
        """Retry rules:
          404 unknown model          -> next fallback model
          429 daily quota            -> next fallback model
          503 overloaded, 4x in a row -> next fallback model
          429 per-minute / 5xx / net -> wait (server hint or exponential) and retry
          other 4xx                  -> fail now (it will never succeed)"""
        delay, overloaded = 4.0, 0
        for attempt in range(tries):
            try:
                if self.provider == "gemini":
                    return self._gemini(prompt, system, want_json)
                if self.provider == "anthropic":
                    return self._anthropic(prompt, system)
                if self.provider in OPENAI_STYLE:
                    return self._openai_style(prompt, system, want_json)
                raise LLMError(f"unknown provider {self.provider!r}")
            except LLMError:
                raise
            except Exception as e:  # network / 429 / 5xx -> back off and retry
                msg = str(e)
                code = (re.match(r"HTTP (\d{3})", msg) or [None, ""])[1]
                if code == "404" and self._switch("model not available (404)"):
                    continue
                if code.startswith("4") and code not in ("408", "429"):
                    raise LLMError(f"{self.provider}:{self.model} rejected the request: {msg[:300]}")
                if code == "429" and re.search(r"PerDay|per day|daily", msg, re.I) and self._switch("daily quota hit"):
                    continue
                if code in ("500", "503", "504"):
                    overloaded += 1
                    if overloaded >= 4 and self._switch(f"overloaded ({code}) 4 times in a row"):
                        overloaded = 0
                        continue
                if attempt == tries - 1:
                    raise LLMError(f"{self.provider} failed after {tries} tries: {msg[:300]}")
                m = re.search(r"retry in ([\d.]+)s|retryDelay['\"]?: ?['\"]?(\d+)", msg)
                sleep = min(float(next(g for g in m.groups() if g)) + 1 if m else delay, 45.0)
                _log(f"[llm] {self.model}: {code or type(e).__name__} -> retry {attempt + 1}/{tries - 1} in {sleep:.0f}s")
                time.sleep(sleep)
                delay = min(delay * 2, 30.0)
        raise LLMError("unreachable")

    def _gemini(self, prompt, system, want_json) -> str:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise LLMError("set GEMINI_API_KEY (free at https://aistudio.google.com/apikey)")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": self.temperature}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        # Gemma 4 thinks for ~3k tokens by default (70 s/call on the free tier). "minimal"
        # gives the same picks in ~8 s. Override with PROMO_THINKING=low|medium|high|off.
        level = os.environ.get("PROMO_THINKING", "minimal" if self.model.startswith("gemma") else "")
        if level and level != "off":
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": level}
        if want_json:
            body["generationConfig"]["responseMimeType"] = "application/json"
        r = requests.post(url, params={"key": key}, json=body, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        um = data.get("usageMetadata", {})
        self.tokens_in += um.get("promptTokenCount", 0)
        self.tokens_out += um.get("candidatesTokenCount", 0)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            # thinking models (Gemma 4, Gemini 3.x) return their reasoning as parts marked
            # thought=true; only the final answer is ours
            answer = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            return answer or "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError):
            raise RuntimeError(f"empty Gemini response: {json.dumps(data)[:300]}")

    def _openai_style(self, prompt, system, want_json) -> str:
        base, key_env = _base_url(self.provider), OPENAI_STYLE[self.provider][1]
        headers = {"Content-Type": "application/json"}
        if key_env:
            key = os.environ.get(key_env)
            if not key:
                raise LLMError(f"set {key_env} for provider {self.provider}")
            headers["Authorization"] = f"Bearer {key}"
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": msgs, "temperature": self.temperature}
        if want_json and self.provider in ("openai", "groq", "ollama"):
            body["response_format"] = {"type": "json_object"}
        r = requests.post(base.rstrip("/") + "/chat/completions", headers=headers, json=body, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        u = data.get("usage") or {}
        self.tokens_in += u.get("prompt_tokens", 0)
        self.tokens_out += u.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"]

    def _anthropic(self, prompt, system) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("set ANTHROPIC_API_KEY")
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=self.timeout,
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": self.model, "max_tokens": 1500, "system": system, "temperature": self.temperature,
                                "messages": [{"role": "user", "content": prompt}]})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        u = data.get("usage", {})
        self.tokens_in += u.get("input_tokens", 0)
        self.tokens_out += u.get("output_tokens", 0)
        return "".join(c.get("text", "") for c in data["content"])


def parse_json(text: str):
    """Models wrap JSON in prose or ``` fences; take the first JSON object/array we can parse."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        s, e = text.find(opener), text.rfind(closer)
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"model did not return JSON: {text[:200]!r}")
