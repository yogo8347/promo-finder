"""Measure the exact token cost of a full run (instead of extrapolating from a sample).

Replays the run from the answer cache (no generation calls, no quota), records every prompt and answer
the pipeline used, and counts their tokens with Gemini's countTokens endpoint (free; it does not use the
generateContent quota).

    python eval/measure_cost.py --input data/input --model gemma-4-26b-a4b-it

Prints input / output tokens per stage and an estimated price. Thinking tokens are not included
(the default thinking level is "minimal").
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from promo_finder import llm as L  # noqa: E402
from promo_finder.pipeline import RunConfig, run  # noqa: E402

CALLS: list[tuple[str, str, str]] = []   # (stage, prompt incl. system, answer)


def stage_of(prompt: str) -> str:
    if "Two candidate promo clips" in prompt:
        return "head-to-head"
    if "Fact-check short descriptions" in prompt:
        return "fact-check"
    if prompt.startswith("Episode ") and "Logline" in prompt:
        return "suggest clips"
    if "Reply with JSON" in prompt and len(prompt) < 80:
        return "pre-flight"
    return "judge"


_orig = L.LLM.complete


def recording_complete(self, prompt, system="", want_json=True):
    text = _orig(self, prompt, system, want_json)
    CALLS.append((stage_of(prompt), (system + "\n\n" + prompt).strip(), text))
    return text


def count(text: str, model: str, key: str) -> int:
    for attempt in range(6):
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens",
                          headers={"x-goog-api-key": key},
                          json={"contents": [{"role": "user", "parts": [{"text": text}]}]}, timeout=60)
        if r.status_code == 200:
            return int(r.json()["totalTokens"])
        time.sleep(2 * (attempt + 1))
    r.raise_for_status()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/input")
    ap.add_argument("--model", default="gemma-4-26b-a4b-it")
    ap.add_argument("--price-in", type=float, default=0.30, help="$ per million input tokens")
    ap.add_argument("--price-out", type=float, default=2.50, help="$ per million output tokens")
    a = ap.parse_args()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set")
    L.LLM.complete = recording_complete
    out_dir = tempfile.mkdtemp(prefix="cost_")
    res = run(RunConfig(input_dir=a.input, out_dir=out_dir, provider="gemini", model=a.model, run_name="cost"),
              progress=lambda m, f: None)
    if res.meta["llm_calls"]:
        print(f"warning: {res.meta['llm_calls']} calls were not cached and ran live")
    tot = defaultdict(lambda: [0, 0, 0])
    for i, (stage, prompt, answer) in enumerate(CALLS, 1):
        t = tot[stage]
        t[0] += 1
        t[1] += count(prompt, a.model, key)
        t[2] += count(answer, a.model, key)
        if i % 25 == 0:
            print(f"  counted {i}/{len(CALLS)}", file=sys.stderr, flush=True)
    rows, ti, to = [], 0, 0
    for stage, (n, i_, o_) in sorted(tot.items(), key=lambda x: -x[1][1]):
        rows.append({"stage": stage, "calls": n, "tokens_in": i_, "tokens_out": o_})
        ti += i_
        to += o_
    cost = ti / 1e6 * a.price_in + to / 1e6 * a.price_out
    summary = {"model": a.model, "calls": len(CALLS), "tokens_in": ti, "tokens_out": to,
               "price_per_million_in_out": [a.price_in, a.price_out], "cost_usd_per_show": round(cost, 3),
               "cost_usd_50_shows": round(cost * 50, 2), "by_stage": rows}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
