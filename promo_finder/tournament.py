"""Head-to-head re-ranking: a Swiss tournament judged by the LLM, fitted with Bradley-Terry.

Why: asked for an absolute 0-10 score, LLM judges are generous and bunch up (in the full run, the top
20 were all "post" and 16 had standalone = 10), so the top of the list is decided by noise. Asked
"which of these two would you post?", the same model has to separate clips. Pairwise comparison is
how human raters are usually made consistent, and it also lets us MEASURE the judge:

* position bias   - how often clip A wins just for being shown first (should be ~50%)
* self-agreement  - re-ask some pairs with A and B swapped; how often the verdict survives

Design
* pool = the top K of the normal ranking (plus any imported clips that would otherwise be left out)
* Swiss rounds: sort by current strength, pair neighbours who have not met yet, so close calls get
  the comparisons; who is shown first alternates so each clip is A about half the time
* Bradley-Terry strengths, refitted after every round (one virtual win + loss per clip as a prior,
  so a 3-0 clip does not shoot to infinity); a confident verdict counts a little more than a toss-up
* the old absolute score is kept as 30% of the blend, so a handful of noisy games can't flip a clear gap
"""
from __future__ import annotations

import math
import random

from .candidates import Promo
from .judge import EDITOR_SYSTEM, _clip_text
from .llm import LLM, LLMError

PAIR_PROMPT = """Two candidate promo clips from the same show, A and B, written out exactly as a viewer would see them \
(time = seconds into the clip).
You are a US viewer scrolling TikTok who has never heard of this show. Pick the ONE clip that would \
(1) stop your scroll in the first 3 seconds, (2) make sense with no context, and (3) cut at a moment that makes you need \
to know what happens next. Ignore which clip is shown first and how long each is.

=== CLIP A ===
{a}

=== CLIP B ===
{b}

Answer JSON only: {{"winner": "A" or "B", "confidence": 1 (coin flip) | 2 (clear) | 3 (no contest), "reason": "<= 20 words"}}"""

CONF_WEIGHT = {1: 0.6, 2: 1.0, 3: 1.4}


def _plain(p: Promo) -> str:
    text, _ = _clip_text(p)
    return "\n".join(line.split(" ", 1)[1] if line.startswith("L") else line for line in text.splitlines())


def bradley_terry(n: int, games: list[tuple[int, int, float]], iters: int = 300) -> list[float]:
    """MM algorithm (Hunter 2004). games = (winner, loser, weight). Returns strengths, geometric mean 1."""
    s = [1.0] * n
    for _ in range(iters):
        new = []
        for i in range(n):
            wins = 1.0 + sum(w for a, _, w in games if a == i)        # +1 virtual win (prior)
            den = 2.0 / (s[i] + 1.0)                                   # virtual win + loss vs strength 1
            for a, b, w in games:
                if i in (a, b):
                    den += w / (s[i] + s[b if a == i else a])
            new.append(wins / den)
        g = math.exp(sum(math.log(x) for x in new) / n)
        s = [x / g for x in new]
    return s


def swiss_pairs(order: list[int], played: set[frozenset]) -> list[tuple[int, int]]:
    pairs, used = [], set()
    for i in order:
        if i in used:
            continue
        for j in order:
            if j != i and j not in used and frozenset((i, j)) not in played:
                pairs.append((i, j))
                used |= {i, j}
                break
    return pairs


def _ask(llm: LLM, a: str, b: str) -> dict | None:
    try:
        r = llm.complete_json(PAIR_PROMPT.format(a=a, b=b), EDITOR_SYSTEM)
    except LLMError:
        return None
    if isinstance(r, list):
        r = r[0] if r else {}
    w = str(r.get("winner", "")).strip().upper()[:1]
    if w not in ("A", "B"):
        return None
    try:
        conf = int(r.get("confidence", 2))
    except (TypeError, ValueError):
        conf = 2
    return {"winner": w, "confidence": min(3, max(1, conf)), "reason": str(r.get("reason", ""))[:160]}


def run_tournament(llm: LLM, pool: list[Promo], rounds: int = 5, recheck: int = 10,
                   label=lambda p: "", progress=None, seed: int = 7) -> dict:
    n = len(pool)
    if n < 4 or not llm.enabled:
        return {"skipped": "needs an LLM and at least 4 clips"}
    texts = [_plain(p) for p in pool]
    games: list[tuple[int, int, float]] = []
    played: set[frozenset] = set()
    as_a = [0] * n
    record = [[0, 0] for _ in range(n)]
    history: list[dict] = []
    a_wins = asked = failed = 0
    strength = [1.0] * n
    # round 1 pairs by the existing order (1v2, 3v4...) - Swiss "seeding"
    order = list(range(n))
    total = rounds * (n // 2) + recheck
    done = 0
    for rnd in range(rounds):
        for i, j in swiss_pairs(order, played):
            a, b = (i, j) if as_a[i] <= as_a[j] else (j, i)
            r = _ask(llm, texts[a], texts[b])
            done += 1
            if progress:
                progress(done, total)
            played.add(frozenset((a, b)))
            if r is None:
                failed += 1
                continue
            as_a[a] += 1
            asked += 1
            win, lose = (a, b) if r["winner"] == "A" else (b, a)
            a_wins += r["winner"] == "A"
            games.append((win, lose, CONF_WEIGHT[r["confidence"]]))
            record[win][0] += 1
            record[lose][1] += 1
            history.append({"round": rnd + 1, "a": a, "b": b, **r})
        strength = bradley_terry(n, games)
        order = sorted(range(n), key=lambda k: -strength[k])

    # self-agreement: replay some decided games with the clips swapped
    rng = random.Random(seed)
    sample = rng.sample(history, min(recheck, len(history)))
    agree = checked = 0
    for h in sample:
        r = _ask(llm, texts[h["b"]], texts[h["a"]])
        done += 1
        if progress:
            progress(done, total)
        if r is None:
            continue
        checked += 1
        first_winner = h["a"] if h["winner"] == "A" else h["b"]
        second_winner = h["b"] if r["winner"] == "A" else h["a"]
        agree += first_winner == second_winner

    logs = [math.log(x) for x in strength]
    lo, hi = min(logs), max(logs)
    for k, p in enumerate(pool):
        p.scores["h2h_wins"], p.scores["h2h_losses"] = record[k]
        p.scores["bt"] = round(logs[k], 3)
        p.scores["tournament"] = round((logs[k] - lo) / (hi - lo), 3) if hi > lo else 0.5
        p.llm["h2h"] = [{"round": h["round"], "vs": label(pool[h["b"] if h["a"] == k else h["a"]]),
                         "won": (h["a"] == k) == (h["winner"] == "A"), "confidence": h["confidence"],
                         "reason": h["reason"]} for h in history if k in (h["a"], h["b"])]
    return {"pool": n, "rounds": rounds, "games": asked, "failed_calls": failed,
            "position_bias_A_win_rate": round(a_wins / asked, 3) if asked else None,
            "swap_recheck": f"{agree}/{checked}" if checked else "n/a",
            "swap_agreement": round(agree / checked, 3) if checked else None}


def blend(pool: list[Promo], w_tournament: float = 0.7) -> None:
    """New final score for the pool: 70% tournament, 30% old score, mapped back onto the pool's own
    score range so the numbers stay comparable with the rest of the list."""
    finals = [p.scores["final"] for p in pool]
    lo, hi = min(finals), max(finals)
    for p in pool:
        p.scores["final_before_h2h"] = p.scores["final"]
        t = lo + (hi - lo) * p.scores.get("tournament", 0.5)
        p.scores["final"] = round((1 - w_tournament) * p.scores["final"] + w_tournament * t, 4)
