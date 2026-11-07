"""Final score, de-duplication, human-readable fields and the posting order."""
from __future__ import annotations

import re

from . import features as F
from .candidates import Promo, _iou
from .parser import fmt


def scene_jumps(p: Promo) -> int:
    """Hard location cuts INSIDE a segment (idea borrowed from the other version's QA): a promo that
    jumps scene mid-clip reads as two clips glued together.  Joins between montage segments are
    deliberate and are not counted."""
    n = 0
    for s in p.segments:
        scenes = [b.scene for b in s.beats if b.kind != "camera" and b.scene]
        n += sum(1 for a, b in zip(scenes, scenes[1:]) if a != b)
    return n


def final_score(p: Promo) -> float:
    rules = p.scores.get("rules", 0.0)
    jump_pen = 1.0 - 0.04 * min(2, scene_jumps(p))
    j = p.llm
    if j and "hook_score" in j:
        llm = (0.35 * j["hook_score"] + 0.35 * j["cliff_score"] + 0.30 * j["standalone_score"]) / 10
        s = 0.35 * rules + 0.65 * llm
        s *= {"post": 1.05, "maybe": 1.0, "reject": 0.7}.get(j.get("verdict", "maybe"), 1.0)
        return round(s * jump_pen, 4)
    return round(rules * jump_pen, 4)


def dedupe(promos: list[Promo]) -> list[Promo]:
    """Same episode and >50% time overlap -> keep the better one."""
    promos = sorted(promos, key=lambda p: -p.scores.get("final", 0))
    kept: list[Promo] = []
    for p in promos:
        if p.kind == "single":
            s = p.segments[0]
            clash = [k for k in kept if k.kind == "single" and k.segments[0].episode == s.episode
                     and _iou((s.start, s.end), (k.segments[0].start, k.segments[0].end)) > 0.5]
            if clash:
                continue
        kept.append(p)
    return kept


def diversify(promos: list[Promo], step: float = 0.04) -> list[Promo]:
    """Greedy re-rank: each extra promo from an episode already in the list loses `step`
    per earlier pick, so one strong episode can't fill the whole top 10.  The raw score is
    kept in scores['final']; the order is what changes."""
    pool = sorted(promos, key=lambda p: -p.scores.get("final", 0))
    out: list[Promo] = []
    seen: dict[int, int] = {}
    while pool:
        best = max(pool, key=lambda p: p.scores["final"] - step * max(seen.get(e, 0) for e in p.episodes))
        pool.remove(best)
        out.append(best)
        for e in set(best.episodes):
            seen[e] = seen.get(e, 0) + 1
    return out


# ------------------------------------------------------------------ text fields ---------
def _describe(b) -> str:
    if b is None:
        return ""
    if b.kind == "dialogue":
        name = " ".join(w[:1].upper() + w[1:].lower() for w in b.speaker.split())
        return f'{name}: "{b.text}"'
    return b.text


def rule_why(p: Promo) -> str:
    """Template pitch when no LLM is used. Built only from signals we actually measured."""
    parts = []
    hb = p.hook_beat
    if hb is not None:
        _, phys = F.physical(hb.text)
        _, shock = F.shock(hb.text)
        if phys:
            parts.append(f"opens on physical shock ({phys[0]})")
        elif shock:
            parts.append(f"opens on a loaded line ({', '.join(shock[:2])})")
        elif hb.tags:
            parts.append(f"opens hot ({', '.join(hb.tags[:2]).lower()})")
    cb = p.cliff_beat
    if cb is not None and cb.kind == "dialogue" and cb.text.strip().endswith("?"):
        parts.append("cuts on an unanswered question")
    elif cb is not None and cb.kind == "dialogue" and cb.text.strip().endswith(("...", "…", "—")):
        parts.append("cuts mid-thought")
    elif any("cliffhanger" in n for n in p.notes):
        parts.append("cuts on the episode's scripted cliffhanger")
    else:
        parts.append("cuts before the payoff")
    text = " ".join(b.text for b in p.beats() if b.kind == "dialogue")
    stakes = sorted({m.group(0).lower() for m in F.UNIVERSAL.finditer(text)})
    if stakes:
        parts.append("stakes a stranger gets instantly: " + ", ".join(stakes[:4]))
    s = "; ".join(parts)
    return s[0].upper() + s[1:] + "."


def fields(p: Promo) -> dict:
    j = p.llm or {}
    prop = j.get("proposal", {})
    hook_rel = None
    if p.hook_beat is not None:
        # when the hook lands = the FIRST beat in the opening 3 s that is nearly as strong as the
        # strongest one (not the strongest itself: a hot line at 0.1 s beats
        # a slightly hotter "No..." at 2.7 s)
        seg = p.segments[0]
        peak = F.beat_intensity(p.hook_beat)
        early = [b for b in seg.beats if b.kind != "camera" and b.start <= seg.start + 3.0
                 and F.beat_intensity(b) >= 0.8 * peak]
        first = min(early, key=lambda b: b.start) if early else p.hook_beat
        hook_rel = max(0.0, first.start - seg.start)
    return {
        "hook": j.get("hook") or prop.get("hook") or _describe(p.hook_beat),
        "hook_at_s": round(hook_rel, 2) if hook_rel is not None else None,
        "cliffhanger": j.get("cliffhanger") or prop.get("cliffhanger") or _describe(p.cliff_beat),
        "why": j.get("why") or prop.get("why") or rule_why(p),
        "source": "llm" if j.get("why") else "rules",
    }


# ------------------------------------------------------------------ posting order -------
def flags(p: Promo) -> list[str]:
    out = []
    f = fields(p)
    if f["hook_at_s"] is not None and f["hook_at_s"] > 2.0:
        out.append(f"hook lands late ({f['hook_at_s']:.1f}s)")
    if p.scores.get("standalone", 1) < 0.6:
        out.append("needs context")
    if max(p.episodes) >= 40:
        out.append("late-story moment (more context, possible spoiler)")
    if p.kind == "montage":
        out.append(f"{len(p.segments) - 1} edit join(s)")
    if scene_jumps(p):
        out.append(f"jumps scene inside the clip ({scene_jumps(p)}x)")
    if p.llm.get("verdict") == "reject":
        out.append("LLM judge rejected")
    if p.llm.get("text_check"):
        out.append("AI description corrected against the script")
    if p.llm.get("confusing"):
        out.append("judge: " + p.llm["confusing"][:80])
    return out


def funnel_bonus(p: Promo) -> float:
    """A new viewer who taps the ad lands on episode 1.  Promos from early episodes sell the
    premise they are about to watch and spoil nothing; late ones need context."""
    last = max(p.episodes)
    if last <= 5:
        return 0.06
    if last <= 15:
        return 0.03
    if last >= 40:
        return -0.03
    return 0.0


def _same_ending(a: Promo, b: Promo, tol: float = 2.0) -> bool:
    x, y = a.segments[-1], b.segments[-1]
    return x.episode == y.episode and abs(x.end - y.end) <= tol


def posting_order(ranked: list[Promo], n: int = 10) -> list[tuple[Promo, str]]:
    pool = sorted(ranked, key=lambda p: -(p.scores["final"] + funnel_bonus(p)))
    plan: list[tuple[Promo, str]] = []
    used_eps: list[set] = []
    while pool and len(plan) < n:
        pick = None
        for p in pool:
            eps = set(p.episodes)
            if plan and eps & used_eps[-1]:
                continue           # don't post two clips of the same scene back to back
            if sum(1 for q, _ in plan if q.kind == "montage") >= 3 and p.kind == "montage":
                continue
            if any(_same_ending(p, q) for q, _ in plan):
                continue           # two posts that cut on the same moment would feel like repeats
            pick = p
            break
        if pick is None:
            pick = pool[0]
        pool.remove(pick)
        reason = []
        if max(pick.episodes) <= 5:
            reason.append("episode 1-5: sells the premise the new viewer will start watching")
        elif funnel_bonus(pick) > 0:
            reason.append("early in the story: little context needed, nothing big spoiled")
        elif funnel_bonus(pick) < 0:
            reason.append("late-story payoff: strong but test after the premise promos")
        reason.append(f"score {pick.scores['final']:.2f}")
        if pick.kind == "montage":
            reason.append("montage gives a fast tour of the arc")
        plan.append((pick, "; ".join(reason)))
        used_eps.append(set(pick.episodes))
    return plan


def segment_str(p: Promo) -> str:
    return " + ".join(f"E{s.episode:02d} {fmt(s.start)}-{fmt(s.end)}" for s in p.segments)


def slug(s: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:n]
