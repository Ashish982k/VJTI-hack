"""OpenAI-compatible LLM client (OpenCode, Groq, ...) with model pools, rate-limit handling and usage."""

import json
import os
import re
import threading
import time

import httpx

from .text import tokens


class LLMError(RuntimeError):
    pass


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


def _env(*names, default=""):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _retry_after(response):
    for header in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        value = response.headers.get(header, "")
        match = re.fullmatch(r"(?:(\d+)m)?([\d.]+)(ms|s)?", value.strip())
        if match:
            seconds = int(match[1] or 0) * 60 + float(match[2]) / (1000 if match[3] == "ms" else 1)
            return min(60.0, max(0.5, seconds))
    return 5.0


class LLMClient:
    """One instance per process. Each call returns (content, usage) so callers can attribute cost.

    LLM_MODEL / LLM_FAST_MODEL may list several comma-separated models: calls rotate through the pool and
    a rate-limited model is skipped until its reset time, so free tiers with per-model limits add up.
    """

    def __init__(self, settings):
        self.settings = settings
        self.key = _env("LLM_API_KEY", "OPENCODE_API_KEY")
        base = _env("LLM_BASE_URL", "OPENCODE_BASE_URL", default="https://opencode.ai/inference/openai/v1")
        base = base.rstrip("/")
        self.endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
        answer = _env("LLM_MODEL", "OPENCODE_MODEL", default="deepseek-v4.1-flash")
        self.answer_models = [m.strip() for m in answer.split(",") if m.strip()]
        fast = _env("LLM_FAST_MODEL", "OPENCODE_FAST_MODEL", default=answer)
        self.fast_models = [m.strip() for m in fast.split(",") if m.strip()]
        self.answer_model = self.answer_models[0]
        # Only sent to models that accept it (e.g. Groq gpt-oss: low/medium/high).
        self.reasoning_effort = _env("LLM_REASONING_EFFORT")
        self.client = httpx.Client(timeout=settings.llm_timeout)
        self.lock = threading.Lock()
        self.turn = 0
        self.cooldown = {}

    @property
    def enabled(self):
        return self.settings.llm_backend != "offline"

    def _pick(self, pool):
        with self.lock:
            now = time.monotonic()
            for offset in range(len(pool)):
                model = pool[(self.turn + offset) % len(pool)]
                if self.cooldown.get(model, 0) <= now:
                    self.turn += offset + 1
                    return model, 0.0
            model = min(pool, key=lambda m: self.cooldown[m])
            return model, self.cooldown[model] - now

    def chat(self, system, user, *, json_mode=False, fast=False, max_tokens=4000, retries=8):
        if not self.key:
            raise LLMError("Set LLM_API_KEY (or OPENCODE_API_KEY), or KB_LLM_BACKEND=offline")
        pool = self.fast_models if fast else self.answer_models
        # Reasoning models spend hidden tokens before the visible reply; budget for both.
        budget = max_tokens + self.settings.reasoning_tokens
        last = None
        for attempt in range(retries):
            model, wait = self._pick(pool)
            if wait > 0:
                time.sleep(wait)
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0,
                "max_tokens": budget,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            if self.reasoning_effort and "gpt-oss" in model:
                payload["reasoning_effort"] = self.reasoning_effort
            started = time.perf_counter()
            try:
                response = self.client.post(
                    self.endpoint, headers={"Authorization": f"Bearer {self.key}"}, json=payload
                )
                if response.status_code == 429:
                    with self.lock:
                        self.cooldown[model] = time.monotonic() + _retry_after(response)
                    raise LLMError(f"{model} rate limited")
                if response.status_code == 413:
                    raise LLMError(f"Request too large for {model}: lower KB_JUDGE_BATCH or KB_CONTEXT_TOKENS")
                if response.status_code in {408, 409} or response.status_code >= 500:
                    raise LLMError(f"Retryable provider status {response.status_code}")
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                finish = result["choices"][0].get("finish_reason")
                if finish == "length" and (json_mode or not (content or "").strip()):
                    # Cut off mid-reasoning or mid-JSON: retrying needs a larger budget, not the same one.
                    budget = min(budget * 2, 32000)
                    raise LLMError(f"{model} output truncated (finish_reason=length)")
                if not isinstance(content, str) or not content.strip():
                    raise LLMError(f"Model returned no content (finish_reason={finish})")
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
                usage = result.get("usage") or {}
                parsed = parse_json(content) if json_mode else content
                return parsed, {
                    "model": model,
                    "input_tokens": usage.get("prompt_tokens", len(tokens(system + user))),
                    "output_tokens": usage.get("completion_tokens", len(tokens(content))),
                    "tokens_estimated": not usage,
                    "seconds": time.perf_counter() - started,
                }
            except (httpx.HTTPError, LLMError, KeyError, ValueError) as e:
                last = e
                if "too large" in str(e):
                    break
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                    break
                if "rate limited" not in str(e):
                    time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"LLM call failed after retries: {last}")


def merge_usage(usages):
    total = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "models": []}
    for usage in usages:
        total["calls"] += 1
        total["input_tokens"] += usage["input_tokens"]
        total["output_tokens"] += usage["output_tokens"]
        if usage["model"] not in total["models"]:
            total["models"].append(usage["model"])
    return total
