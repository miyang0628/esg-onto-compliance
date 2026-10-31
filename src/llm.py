"""
llm.py — shared OpenAI judge: cached llm_call() + CostTracker + model tiers.

Extracted from notebook 05 Cell A so that BOTH the cross-lingual judgment
(nb05) and the evidence-conditioned judgment / bottleneck experiment (nb09)
use the *identical* call path and cache. This matters for the revision:
reviewer 2 (M2) asks us to compare judgments on the same cases under
different evidence; using one judge implementation guarantees the only thing
that changes across conditions is the evidence, not the plumbing.

Caching
-------
Responses are cached under artifacts/llm_cache/<key>.json where key hashes
(model, system, user, temperature, response_json). Re-running is therefore
free for already-seen prompts. The shipped cache covers the WEAK tier only;
mid/strong tiers and any new (evidence-conditioned) prompt will issue live
calls on the user's machine with their own OPENAI_API_KEY.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional; env vars may be set another way
    def load_dotenv(*_a, **_k):  # type: ignore
        return False

from config import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "artifacts" / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# model tiers (names come from .env; pricing from the user's manifest)
# --------------------------------------------------------------------------
def load_models() -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(Path.cwd() / ".env")
    return {
        "weak":   {"name": os.getenv("LLM_MODEL_WEAK", "gpt-4o-mini"),
                   "in": 0.15, "cached_in": 0.075, "out": 0.60},
        "mid":    {"name": os.getenv("LLM_MODEL_MID", "gpt-5.4-mini"),
                   "in": 0.75, "cached_in": 0.075, "out": 4.50},
        "strong": {"name": os.getenv("LLM_MODEL_STRONG", "gpt-5.4"),
                   "in": None, "cached_in": None, "out": None},
    }


MODELS = load_models()


class CostTracker:
    def __init__(self):
        self.records: list[dict] = []

    def add(self, tier, model, usage, tag=""):
        p = MODELS.get(tier, {})
        pin, pcached, pout = p.get("in"), p.get("cached_in"), p.get("out")
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        cached = 0
        det = getattr(usage, "prompt_tokens_details", None)
        if det is not None:
            cached = getattr(det, "cached_tokens", 0) or 0
        fresh = max(pt - cached, 0)
        cost = None
        if None not in (pin, pout):
            pc = pcached if pcached is not None else pin
            cost = (fresh * pin + cached * pc + ct * pout) / 1_000_000
        self.records.append({"tag": tag, "tier": tier, "model": model,
                             "prompt_tokens": pt, "cached_tokens": cached,
                             "completion_tokens": ct, "cost_usd": cost})
        return cost or 0.0

    def total_usd(self):
        return float(sum(r["cost_usd"] or 0.0 for r in self.records))


COST = CostTracker()


def _safe_json(text):
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = t.find(opener), t.rfind(closer)
        if 0 <= i < j:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                continue
    return None


def _cache_key(model, system, user, temperature, response_json):
    raw = json.dumps({"m": model, "s": system, "u": user,
                      "t": temperature, "j": response_json},
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _client():
    """Lazy OpenAI client so that offline, cache-only runs need no API key."""
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set (needed only for cache misses).")
    return OpenAI(api_key=key)


def llm_call(system, user, tier="weak", temperature=0.0,
             response_json=True, tag="", use_cache=True, max_retries=3):
    """Cached chat call. Returns {text, json, cached, tier, model, cost_usd}.

    Identical key/behaviour to the submitted nb05 so the shipped cache keeps
    hitting. A cache MISS triggers a live call (needs OPENAI_API_KEY).
    """
    model = MODELS[tier]["name"]
    key = _cache_key(model, system, user, temperature, response_json)
    cache_file = CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        c = json.loads(cache_file.read_text(encoding="utf-8"))
        c["cached"] = True
        original_cost = c.get("cost_usd", 0.0) or 0.0
        COST.records.append({"tag": tag or c.get("model", ""),
                             "tier": c.get("tier", tier), "model": c.get("model", ""),
                             "prompt_tokens": 0, "cached_tokens": 0,
                             "completion_tokens": 0, "cost_usd": original_cost})
        c["cost_usd"] = original_cost
        return c

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    if response_json:
        kwargs["response_format"] = {"type": "json_object"}
    kwargs["temperature"] = temperature

    client = _client()
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            parsed = _safe_json(text) if response_json else None
            if response_json and parsed is None:
                raise ValueError("Response was not valid JSON.")
            cost = COST.add(tier, model, resp.usage, tag=tag or model)
            out = {"text": text, "json": parsed, "cached": False,
                   "tier": tier, "model": model, "cost_usd": cost}
            if use_cache:
                cache_file.write_text(json.dumps(out, ensure_ascii=False),
                                      encoding="utf-8")
            return out
        except TypeError as e:
            if "temperature" in kwargs:
                kwargs.pop("temperature", None)
                last_err = e
                continue
            last_err = e
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"[llm_call] failed after {max_retries} retries: {last_err}")


def tier_available(tier) -> bool:
    """Cheap probe: try a 1-token cached/real call; False if unreachable."""
    try:
        llm_call("ping", 'Return JSON {"ok":true}.', tier=tier, tag=f"probe-{tier}")
        return True
    except Exception as e:
        print(f"[skip] tier '{tier}' ({MODELS[tier]['name']}) unavailable: {e}")
        return False
