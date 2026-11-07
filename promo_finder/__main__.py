"""CLI:  python -m promo_finder --input data/input [--provider gemini] [--model gemini-2.5-flash]"""
from __future__ import annotations

import argparse
import sys

from .pipeline import RunConfig, run


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="promo_finder", description="Find ranked 20-60s promos in timestamped microdrama scripts.")
    ap.add_argument("--input", default="data/input", help="folder of episode .txt screenplays")
    ap.add_argument("--out", default="outputs", help="where run folders are written")
    ap.add_argument("--provider", default=None,
                    help="none | gemini | groq | openrouter | ollama | openai | anthropic (default: $PROMO_LLM_PROVIDER or none)")
    ap.add_argument("--model", default=None, help="model name for the provider (default: provider's default)")
    ap.add_argument("--judge-top", type=int, default=60, help="how many rule candidates the LLM judges")
    ap.add_argument("--judge-batch", type=int, default=3, help="clips per judge call (1 = one call per clip)")
    ap.add_argument("--no-propose", action="store_true", help="skip the LLM reading whole episodes")
    ap.add_argument("--per-episode", type=int, default=4)
    ap.add_argument("--montages", type=int, default=12)
    ap.add_argument("--top", type=int, default=30, help="promos written to promos.md")
    ap.add_argument("--name", default="", help="run folder name")
    ap.add_argument("--merge", action="append", default=[], metavar="SLATE.json",
                    help="import another tool's promo slate; its clips are re-cut to legal cuts and re-judged blind (repeatable)")
    ap.add_argument("--tournament", type=int, default=28, help="head-to-head pool size (0 = off; needs an LLM)")
    ap.add_argument("--rounds", type=int, default=5, help="Swiss rounds in the head-to-head")
    ap.add_argument("--recheck", type=int, default=10, help="games replayed with A/B swapped to measure the judge")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    res = run(RunConfig(input_dir=a.input, out_dir=a.out, provider=a.provider, model=a.model,
                        judge_top=a.judge_top, judge_batch=a.judge_batch, llm_propose=not a.no_propose, per_episode=a.per_episode,
                        montages=a.montages, top_n=a.top, run_name=a.name,
                        merge=a.merge, tournament=a.tournament, rounds=a.rounds, recheck=a.recheck))
    print()
    print(f"engine: {res.meta['engine']}   promos: {len(res.ranked)}   validator failures: {len(res.qa['validator_failures'])}")
    rec = (res.qa.get("recall") or {}).get("recall", {})
    if rec:
        print("reference recall:", ", ".join(f"{k} {v['all_refs']} (A {v['tier_A']})" for k, v in rec.items()))
    print("\nposting order:")
    from .rank import fields, segment_str
    for n, (p, why) in enumerate(res.plan[:10], 1):
        print(f" {n:>2}. {p.id} {segment_str(p):<45} {p.duration:4.0f}s  {fields(p)['hook'][:60]}")
    print(f"\nwritten to {res.out_path}")


if __name__ == "__main__":
    main()
