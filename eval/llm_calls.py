"""
LLM call layer for the Phase-2 experiments (self-repair, ablation, repeated sampling).

Differences from the private callers in converter/llm_converter.py:
  * captures token usage and cost per call;
  * a hard call budget (CallBudget) and a dry-run mode in which NO LLM is called
    (the "response" is the text of a caller-supplied file, so the whole
    generate -> validate -> record pipeline can be exercised at zero cost);
  * the Claude CLI is invoked with --output-format json (usage + cost) and
    --tools "" so the model cannot read repository files during generation.

Nothing here modifies converter/llm_converter.py; the prompt texts used by the
experiments are imported from it so they stay byte-identical to the archived
benchmark prompts.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from autofl.converter.llm_converter import (
    CLAUDE_MODEL, GEMINI_MODEL, OLLAMA_MODEL, _strip_fences,
)

# USD per 1M tokens, used only for providers that do not report cost themselves.
# Check the provider price page before reporting; PRICE_VERSION is written to every row.
PRICE_VERSION = "2026-09-22"
PRICES = {
    GEMINI_MODEL: {"input": 0.30, "output": 2.50},   # output price also applied to thinking tokens
}


class BudgetExceeded(RuntimeError):
    pass


class CallBudget:
    """Counts live LLM calls and raises once max_calls is exceeded (None = unlimited)."""

    def __init__(self, max_calls: int | None):
        self.max_calls = max_calls
        self.calls = 0

    def tick(self) -> None:
        self.calls += 1
        if self.max_calls is not None and self.calls > self.max_calls:
            raise BudgetExceeded(f"live LLM call budget of {self.max_calls} exceeded")


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = -1
    output_tokens: int = -1
    thinking_tokens: int = 0
    cost_usd: float = float("nan")
    cost_source: str = ""          # provider | price_table | n/a | dry_run
    elapsed_sec: float = 0.0
    temperature: str = ""          # "cli-default" for the Claude CLI
    dry_run: bool = False
    raw_usage: dict = field(default_factory=dict)

    def usage_dict(self) -> dict:
        d = asdict(self)
        d.pop("text")
        d.pop("raw_usage")
        d["price_version"] = PRICE_VERSION
        return {f"llm_{k}": v for k, v in d.items()}   # prefixed: never collides with eval columns


# ── providers ────────────────────────────────────────────────────────────────

# The CLI is run from an empty directory so that no project files, CLAUDE.md or
# settings can enter the context (in addition to --tools "" which disables file access).
CLAUDE_CWD = Path(os.environ.get("AUTOFL_CLAUDE_CWD", "/tmp/autofl_claude_cli_cwd"))


def _call_claude(system: str | None, user: str) -> LLMResponse:
    CLAUDE_CWD.mkdir(parents=True, exist_ok=True)
    cmd = [
        "claude", "-p",
        "--model", CLAUDE_MODEL,
        "--output-format", "json",
        "--tools", "",
        "--no-session-persistence",
    ]
    if system:
        cmd += ["--system-prompt", system]
    t0 = time.time()
    r = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=900, cwd=str(CLAUDE_CWD))
    el = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"claude CLI failed (rc={r.returncode}): {r.stderr[:800]}")
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"claude CLI returned non-JSON output: {r.stdout[:300]}") from e
    if data.get("is_error"):
        raise RuntimeError(f"claude CLI reported error: {str(data.get('result'))[:500]}")
    usage = data.get("usage") or {}
    inp = int(usage.get("input_tokens", 0) or 0) \
        + int(usage.get("cache_creation_input_tokens", 0) or 0) \
        + int(usage.get("cache_read_input_tokens", 0) or 0)
    out = int(usage.get("output_tokens", 0) or 0)
    cost = data.get("total_cost_usd")
    model_used = CLAUDE_MODEL
    mu = data.get("modelUsage") or {}
    if isinstance(mu, dict) and mu:
        model_used = next(iter(mu.keys()))
    return LLMResponse(
        text=data.get("result", "") or "",
        provider="claude", model=model_used,
        input_tokens=inp, output_tokens=out,
        cost_usd=float(cost) if cost is not None else float("nan"),
        cost_source="provider" if cost is not None else "n/a",
        elapsed_sec=round(el, 2), temperature="cli-default",
        raw_usage={"usage": usage, "modelUsage": mu, "duration_ms": data.get("duration_ms")},
    )


def _call_gemini(system: str | None, user: str, temperature: float | None) -> LLMResponse:
    from google import genai
    from google.genai import types
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY / GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(
        system_instruction=system if system else None,
        temperature=temperature,
    )
    t0 = time.time()
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=user, config=cfg)
    el = time.time() - t0
    um = resp.usage_metadata
    inp = int(getattr(um, "prompt_token_count", 0) or 0)
    out = int(getattr(um, "candidates_token_count", 0) or 0)
    th = int(getattr(um, "thoughts_token_count", 0) or 0)
    price = PRICES.get(GEMINI_MODEL)
    cost = (inp * price["input"] + (out + th) * price["output"]) / 1e6 if price else float("nan")
    return LLMResponse(
        text=resp.text or "", provider="gemini", model=GEMINI_MODEL,
        input_tokens=inp, output_tokens=out, thinking_tokens=th,
        cost_usd=cost, cost_source="price_table" if price else "n/a",
        elapsed_sec=round(el, 2),
        temperature="provider-default" if temperature is None else str(temperature),
        raw_usage={"prompt": inp, "candidates": out, "thoughts": th,
                   "total": getattr(um, "total_token_count", None)},
    )


def _call_ollama(system: str | None, user: str, temperature: float | None) -> LLMResponse:
    import ollama
    model_name = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    opts = {} if temperature is None else {"temperature": temperature}
    t0 = time.time()
    resp = ollama.chat(model=model_name, messages=messages, options=opts)
    el = time.time() - t0
    return LLMResponse(
        text=resp["message"]["content"], provider="ollama", model=model_name,
        input_tokens=int(resp.get("prompt_eval_count", -1) or -1),
        output_tokens=int(resp.get("eval_count", -1) or -1),
        cost_usd=0.0, cost_source="local", elapsed_sec=round(el, 2),
        temperature="provider-default" if temperature is None else str(temperature),
    )


# ── public API ───────────────────────────────────────────────────────────────

RETRY_SLEEP_SEC = (60, 180, 600)   # transient failures: rate limits, timeouts, 5xx


class QuotaExhausted(RuntimeError):
    """A per-day provider quota: retrying within the same day cannot succeed."""


_DAILY_QUOTA_RE = re.compile(r"PerDay|GenerateRequestsPerDay|per day", re.I)


def _is_daily_quota(e: Exception) -> bool:
    m = str(e)
    return "RESOURCE_EXHAUSTED" in m and bool(_DAILY_QUOTA_RE.search(m))


def _with_retry(fn, *args):
    last = None
    for i, pause in enumerate((0,) + RETRY_SLEEP_SEC):
        if pause:
            print(f"    [retry {i}/{len(RETRY_SLEEP_SEC)} after {pause}s: {str(last)[:120]}]", flush=True)
            time.sleep(pause)
        try:
            return fn(*args)
        except (subprocess.TimeoutExpired, RuntimeError, OSError) as e:   # noqa: PERF203
            last = e
        except Exception as e:                                             # provider SDK errors
            last = e
        if _is_daily_quota(last):        # sleeping minutes cannot clear a per-day quota
            raise QuotaExhausted(str(last)[:300]) from last
    raise RuntimeError(f"LLM call failed after retries: {last}")


def call_llm(
    provider: str,
    system: str | None,
    user: str,
    *,
    temperature: float | None = None,
    budget: CallBudget | None = None,
    dry_run_source: str | Path | None = None,
) -> LLMResponse:
    """One LLM call. With dry_run_source set, no LLM is called and the file's
    text is returned as the response (pipeline test)."""
    if dry_run_source is not None:
        text = Path(dry_run_source).read_text()
        return LLMResponse(text=text, provider=provider, model="dry-run",
                           input_tokens=len(user) // 4 + (len(system) // 4 if system else 0),
                           output_tokens=len(text) // 4, cost_usd=0.0,
                           cost_source="dry_run", dry_run=True, temperature="n/a")
    if budget is not None:
        budget.tick()
    if provider == "claude":
        return _with_retry(_call_claude, system, user)
    if provider == "gemini":
        return _with_retry(_call_gemini, system, user, temperature)
    if provider == "ollama":
        return _with_retry(_call_ollama, system, user, temperature)
    raise ValueError(f"unknown provider {provider!r}")


_FENCE_RE = re.compile(r"```(?:python|py|python3)?[ \t]*\r?\n(.*?)```", re.S)


def extract_code(raw: str) -> str:
    """Return the Python module contained in an LLM response.

    The archived v1 stripper (converter._strip_fences) only removes fences when the
    response STARTS with a fence, so a response that opens with one line of prose
    followed by a fenced module was stored verbatim and failed the syntax check.
    Here the largest fenced block is taken when any fence is present; a response
    without fences is returned unchanged.
    """
    raw = raw.strip()
    blocks = _FENCE_RE.findall(raw)
    if blocks:
        return max(blocks, key=len).strip()
    if raw.startswith("```"):                       # unterminated fence
        raw = "\n".join(raw.splitlines()[1:])
    # stray fence lines (e.g. a closing ``` with no opening one) are dropped
    lines = [l for l in raw.splitlines() if not re.fullmatch(r"\s*```[A-Za-z0-9]*\s*", l)]
    return "\n".join(lines).strip()


def generate_file(provider: str, system: str | None, user: str, out_path: str | Path,
                  reuse_existing: bool = False, **kw) -> tuple[Path, LLMResponse]:
    """Call the LLM, save the raw response next to the module (<out>.raw.txt),
    extract the code, write the module to out_path. With reuse_existing, a
    module (or raw response) already on disk is reused without a new call;
    its usage fields are then unknown (cost_source = "reused_file")."""
    out = Path(out_path)
    raw_p = out.with_suffix(out.suffix + ".raw.txt")
    if reuse_existing and (raw_p.exists() or out.exists()) and not kw.get("dry_run_source"):
        text = raw_p.read_text() if raw_p.exists() else out.read_text()
        out.write_text(extract_code(text))
        return out, LLMResponse(text=text, provider=provider, model="reused", cost_usd=float("nan"),
                                cost_source="reused_file", temperature="n/a")
    resp = call_llm(provider, system, user, **kw)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not resp.dry_run:
        out.with_suffix(out.suffix + ".raw.txt").write_text(resp.text)
    out.write_text(extract_code(resp.text))
    return out, resp
