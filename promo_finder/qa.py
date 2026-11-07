"""Quality checks.

1. Hard validators - every output promo must pass, or the run says so loudly.
2. Recall against reference picks - does the tool surface what a human editor would pick?
3. Failure listing - the weakest spots in the top of the ranking, for a human to look at.
"""
from __future__ import annotations

import json
from pathlib import Path

from .candidates import MAX_LEN, MIN_LEN, Promo
from .parser import Episode, fmt, to_seconds


def validate(p: Promo, eps: dict[int, Episode]) -> list[str]:
    errs = []
    if not (MIN_LEN - 0.01 <= p.duration <= MAX_LEN + 0.01):
        errs.append(f"duration {p.duration:.1f}s outside 20-60s")
    prev = (-1, -1.0)
    for s in p.segments:
        ep = eps.get(s.episode)
        if ep is None:
            errs.append(f"unknown episode {s.episode}")
            continue
        if not (0 <= s.start < s.end <= ep.duration + 0.01):
            errs.append(f"E{s.episode} {fmt(s.start)}-{fmt(s.end)} outside the episode")
        if (s.episode, s.start) <= prev:
            errs.append("segments not in story order")
        prev = (s.episode, s.start)
        for t in (s.start, s.end):
            for b in ep.beats:
                if b.kind == "dialogue" and b.start < t - 0.25 and b.end > t + 0.25:
                    errs.append(f"cut at E{s.episode} {fmt(t)} lands inside a line: {b.text[:40]!r}")
    if p.hook_beat is None:
        errs.append("no hook beat")
    elif p.hook_beat.start - p.segments[0].start > 3.0 + 1e-6:
        errs.append(f"hook at {p.hook_beat.start - p.segments[0].start:.1f}s, after the 3s window")
    return errs


def _overlap_frac(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / max(1e-6, min(a1 - a0, b1 - b0))


def matches(p: Promo, ref: dict) -> bool:
    r0, r1 = to_seconds(ref["start"]), to_seconds(ref["end"])
    return any(s.episode == ref["episode"] and _overlap_frac(s.start, s.end, r0, r1) >= 0.5 for s in p.segments)


def recall_report(ranked: list[Promo], ref_path: str | Path, eps: dict[int, Episode] | None = None) -> dict:
    """Only runs when the reference file belongs to this show (its fingerprint must appear in
    the first episode's logline), so a new show is never scored against the wrong picks."""
    path = Path(ref_path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    fp = data.get("fingerprint", "").lower()
    if eps and fp:
        first = eps[min(eps)]
        if fp not in (first.logline + " " + first.continuity).lower():
            return {"skipped": f"reference picks are for '{data.get('show', '?')}', not this show"}
    picks = data["picks"]
    out = {"n_reference": len(picks), "recall": {}, "per_pick": []}
    for k in (10, 20, 30, len(ranked)):
        top = ranked[:k]
        hit = sum(1 for r in picks if any(matches(p, r) for p in top))
        hitA = sum(1 for r in picks if r["tier"] == "A" and any(matches(p, r) for p in top))
        nA = sum(1 for r in picks if r["tier"] == "A")
        out["recall"][f"top{k}" if k != len(ranked) else "all"] = {
            "all_refs": f"{hit}/{len(picks)}", "tier_A": f"{hitA}/{nA}"}
    top20 = ranked[:20]
    out["top20_matching_a_reference"] = f"{sum(1 for p in top20 if any(matches(p, r) for r in picks))}/{len(top20)}"
    for r in picks:
        rank = next((i + 1 for i, p in enumerate(ranked) if matches(p, r)), None)
        out["per_pick"].append({"id": r["id"], "episode": r["episode"], "tier": r["tier"],
                                "moment": r["moment"], "found_at_rank": rank})
    return out
