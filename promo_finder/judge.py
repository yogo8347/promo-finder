"""LLM steps.  Both are optional; the pipeline works without them.

propose(episode)  the model reads a whole episode and suggests promos by LINE ID.
judge(promo)      the model watches ONLY the clip (blind, like a stranger scrolling TikTok)
                  and scores hook / cliffhanger / standalone, writes the one-line pitch,
                  and may suggest a tighter cut, again by line ID.

The model never writes a timestamp.  It points at numbered lines; code maps them to the
script's own timestamps and re-checks every rule (20-60 s, hook inside 3 s, no mid-line
cut).  Anything that fails is dropped and logged, never "fixed" silently.
"""
from __future__ import annotations

from .candidates import MIN_LEN, MAX_LEN, Promo, Segment, end_points, score_window, start_points
from .llm import LLM, LLMError
from .parser import Episode, fmt

EDITOR_SYSTEM = """You are a senior performance-marketing editor for vertical microdrama apps \
(ReelShort / DramaBox style). You cut 20-60 second promo clips from the show itself for \
TikTok, Instagram Reels and YouTube Shorts, aimed at US viewers who have never seen the show.
A great promo:
- HOOK: something in the first 3 seconds stops the scroll (a slap, a public humiliation, a reveal, a shocking line). Slow establishing shots, soft talk or a line that starts mid-sentence are bad hooks.
- CLIFFHANGER: it cuts at an unresolved moment (a question hangs, a threat lands, a door bursts open, a reveal is half-made), so the viewer must download the app to see what happens next. Ending after the conflict is resolved is bad.
- STANDALONE: a stranger understands who is wronged and what is at stake from the clip alone. Unexplained names, lore and callbacks hurt.
You are strict: most clips are 4-7 out of 10. Reserve 9-10 for clips you would put ad budget behind today.
Answer with JSON only."""


def _numbered(beats, t0: float = 0.0, with_camera: bool = False) -> tuple[str, dict]:
    """Lines the model can point at: L1, L2, ...  Returns text and id -> beat."""
    rows, ids = [], {}
    n = 0
    for b in beats:
        if b.kind == "camera" and not with_camera:
            continue
        n += 1
        ids[f"L{n}"] = b
        rows.append(f"L{n} [{fmt(b.start - t0)}] {b.label()}")
    return "\n".join(rows), ids


# ------------------------------------------------------------------ judge ---------------
JUDGE_PROMPT = """Here is a candidate promo clip, exactly as a viewer would see it (time = seconds into the promo).
Lines marked "---" are hard cuts between segments.

{clip}

Score it and return JSON with exactly these keys:
{{
 "hook_score": 0-10,
 "hook": "what grabs the viewer in the first 3 seconds (quote or describe it)",
 "cliffhanger_score": 0-10,
 "cliffhanger": "the unresolved moment the clip cuts on",
 "standalone_score": 0-10,
 "confusing_for_new_viewer": "what a stranger would not understand, or empty string",
 "why_it_works": "ONE line pitch: why this promo would make someone download the app",
 "verdict": "post" | "maybe" | "reject",
 "better_start": "line id (e.g. L3) to start on instead, or null if the start is right",
 "better_end": "line id to end on instead, or null if the end is right"
}}"""


def _clip_text(p: Promo) -> tuple[str, dict]:
    rows, ids, offset, n = [], {}, 0.0, 0
    for k, s in enumerate(p.segments):
        if len(p.segments) > 1:
            rows.append(f"--- cut to segment {k + 1} ---")
        for b in s.beats:
            if b.kind == "camera":
                continue
            n += 1
            ids[f"L{n}"] = (k, b)
            rows.append(f"L{n} [{fmt(offset + b.start - s.start)}] {b.label()}")
        offset += s.dur
    return "\n".join(rows), ids


def judge(llm: LLM, p: Promo, eps: dict[int, Episode]) -> None:
    """Fills p.llm in place. Applies a suggested tighter cut only if it passes every rule."""
    clip, ids = _clip_text(p)
    try:
        r = llm.complete_json(JUDGE_PROMPT.format(clip=clip), EDITOR_SYSTEM)
    except LLMError as e:
        p.llm = {**p.llm, "error": str(e)[:200]}
        return
    _apply_verdict(p, r, ids, eps)


BATCH_PROMPT = """Below are {n} unrelated candidate promo clips (C1..C{n}). Judge EACH ONE ON ITS OWN, \
as if it were the only clip you saw: never compare clips, never mention another clip, and never use \
information from one clip to understand another.
Time = seconds into that promo. Lines marked "---" are hard cuts between segments. Line ids restart in every clip.

{clips}

Return JSON: {{"clips": [ one object per clip, in order, each with exactly these keys:
 "clip": "C1",
 "hook_score": 0-10,
 "hook": "what grabs the viewer in the first 3 seconds (quote or describe it)",
 "cliffhanger_score": 0-10,
 "cliffhanger": "the unresolved moment the clip cuts on",
 "standalone_score": 0-10,
 "confusing_for_new_viewer": "what a stranger would not understand, or empty string",
 "why_it_works": "ONE line pitch: why this promo would make someone download the app",
 "verdict": "post" | "maybe" | "reject",
 "better_start": "line id inside THIS clip to start on instead, or null",
 "better_end": "line id inside THIS clip to end on instead, or null"
]}}"""


def judge_many(llm: LLM, promos: list[Promo], eps: dict[int, Episode], batch: int = 5,
               progress=None) -> list[str]:
    """Judge clips in batches (5x fewer calls: matters on free tiers). A batch whose answer
    can't be matched clip-by-clip is retried one clip at a time, so nothing is mis-assigned."""
    log: list[str] = []
    done = 0
    for group in _spread_batches(promos, batch):
        texts, idmaps = [], []
        for k, p in enumerate(group, 1):
            t, ids = _clip_text(p)
            texts.append(f"=== C{k} ===\n{t}")
            idmaps.append(ids)
        ok = False
        try:
            r = llm.complete_json(BATCH_PROMPT.format(n=len(group), clips="\n\n".join(texts)), EDITOR_SYSTEM)
            items = r.get("clips") if isinstance(r, dict) else r
            if isinstance(items, list) and len(items) == len(group):
                by_label = {str(x.get("clip", "")).strip().upper(): x for x in items if isinstance(x, dict)}
                for k, p in enumerate(group, 1):
                    x = by_label.get(f"C{k}") or (items[k - 1] if isinstance(items[k - 1], dict) else None)
                    if x is None:
                        raise LLMError("missing clip")
                    _apply_verdict(p, x, idmaps[k - 1], eps)
                ok = True
            else:
                log.append(f"batch of {len(group)} returned {len(items) if isinstance(items, list) else 'no'} verdicts; judging one by one")
        except LLMError as e:
            log.append(f"batch judge failed ({str(e)[:120]}); judging one by one")
        if not ok:
            for p in group:
                judge(llm, p, eps)
        done += len(group)
        if progress:
            progress(done, len(promos))
    return log


def _spread_batches(promos: list[Promo], size: int) -> list[list[Promo]]:
    """Group clips so no batch holds two clips that share an episode. Overlapping cuts of the
    same scene side by side make the model compare them ("starts slower than C1") instead of
    judging each blind, which we saw in a trial run."""
    if size <= 1:
        return [[p] for p in promos]
    batches: list[list[Promo]] = []
    for p in promos:
        eps = set(p.episodes)
        for b in batches:
            if len(b) < size and not any(eps & set(q.episodes) for q in b):
                b.append(p)
                break
        else:
            batches.append([p])
    return batches


def _apply_verdict(p: Promo, r, ids: dict, eps: dict[int, Episode]) -> None:
    if not isinstance(r, dict):
        p.llm = {**p.llm, "error": "non-object JSON"}
        return

    def num(k):
        try:
            return max(0.0, min(10.0, float(r.get(k, 0))))
        except (TypeError, ValueError):
            return 0.0

    p.llm = {
        **p.llm,   # keep the proposer's notes if this promo came from propose()
        "hook_score": num("hook_score"), "cliff_score": num("cliffhanger_score"),
        "standalone_score": num("standalone_score"),
        "hook": str(r.get("hook", ""))[:300], "cliffhanger": str(r.get("cliffhanger", ""))[:300],
        "confusing": str(r.get("confusing_for_new_viewer", ""))[:300],
        "why": str(r.get("why_it_works", ""))[:300],
        "verdict": str(r.get("verdict", "maybe")).lower().strip(),
    }
    # optional re-cut (single-segment promos only; montages are left to the editor)
    if p.kind == "single":
        bs, be = r.get("better_start"), r.get("better_end")
        if bs or be:
            _try_recut(p, ids, bs, be, eps[p.segments[0].episode])


def _try_recut(p: Promo, ids: dict, bs, be, ep: Episode) -> None:
    seg = p.segments[0]
    t0, t1 = seg.start, seg.end
    if bs and str(bs) in ids:
        b = ids[str(bs)][1]
        cands = [s for s in start_points(ep) if s <= b.start + 1e-6 and b.start - s <= 1.5]
        if cands:
            t0 = max(cands)
    if be and str(be) in ids:
        b = ids[str(be)][1]
        cands = [e for e in end_points(ep) if e >= b.end - 1e-6 and e - b.end <= 2.0]
        if cands:
            t1 = min(cands)
    if (t0, t1) == (seg.start, seg.end):
        return
    if not (MIN_LEN <= t1 - t0 <= MAX_LEN):
        p.notes.append(f"LLM re-cut {fmt(t0)}-{fmt(t1)} rejected: {t1 - t0:.1f}s is outside 20-60s")
        return
    sc = score_window(ep, t0, t1)
    if sc["_hb"] is None:
        p.notes.append("LLM re-cut rejected: no hook in first 3s")
        return
    p.notes.append(f"re-cut by LLM from {fmt(seg.start)}-{fmt(seg.end)}")
    p.segments = [Segment(ep.number, t0, t1, sc["_beats"])]
    p.hook_beat, p.cliff_beat = sc["_hb"], sc["_cb"]
    p.scores.update({k: round(v, 3) for k, v in sc.items() if not k.startswith("_")})


# ------------------------------------------------------------------ propose -------------
PROPOSE_PROMPT = """Episode {num} of a vertical microdrama. Logline: {logline}

Every line has an id and its timestamp inside the episode:
{lines}

Find up to 3 of the best 20-60 second promo clips in this episode (one continuous stretch each).
Each must open on a hook within its first 3 seconds and cut on a cliffhanger.
Return JSON: {{"promos": [{{"start": "L?", "end": "L?", "hook": "...", "cliffhanger": "...", "why": "one line"}}]}}
Use only line ids that appear above. If nothing in this episode is promo-worthy, return {{"promos": []}}."""


def propose(llm: LLM, ep: Episode) -> tuple[list[Promo], list[str]]:
    lines, ids = _numbered(ep.beats)
    log: list[str] = []
    try:
        r = llm.complete_json(PROPOSE_PROMPT.format(num=ep.number, logline=ep.logline, lines=lines), EDITOR_SYSTEM)
    except LLMError as e:
        return [], [f"E{ep.number}: propose failed: {str(e)[:150]}"]
    items = r.get("promos", []) if isinstance(r, dict) else []
    out: list[Promo] = []
    sp, epts = start_points(ep), end_points(ep)
    for it in items[:3]:
        if not isinstance(it, dict):
            continue
        s_id, e_id = str(it.get("start")), str(it.get("end"))
        if s_id not in ids or e_id not in ids:
            log.append(f"E{ep.number}: dropped proposal with unknown line ids {s_id}-{e_id} (hallucinated id)")
            continue
        bs, be = ids[s_id], ids[e_id]
        s_c = [s for s in sp if s <= bs.start + 1e-6 and bs.start - s <= 1.5]
        e_c = [e for e in epts if e >= be.end - 1e-6 and e - be.end <= 2.0]
        if not s_c or not e_c:
            log.append(f"E{ep.number}: dropped {s_id}-{e_id}: no clean cut point near those lines")
            continue
        t0, t1 = max(s_c), min(e_c)
        repaired = ""
        if not (MIN_LEN <= t1 - t0 <= MAX_LEN):
            fixed = _repair_length(ep, t0, t1, sp, epts)
            if fixed is None:
                log.append(f"E{ep.number}: dropped {s_id}-{e_id}: {t1 - t0:.1f}s outside 20-60s, no valid repair")
                continue
            repaired = (f"model picked {fmt(t0)}-{fmt(t1)} ({t1 - t0:.1f}s); code {fixed[2]} to "
                        f"{fmt(fixed[0])}-{fmt(fixed[1])} ({fixed[1] - fixed[0]:.1f}s)")
            log.append(f"E{ep.number}: repaired {s_id}-{e_id}: {repaired}")
            t0, t1 = fixed[0], fixed[1]
        sc = score_window(ep, t0, t1)
        if sc["_hb"] is None:
            log.append(f"E{ep.number}: dropped {s_id}-{e_id}: dead air in first 3s")
            continue
        p = Promo("single", [Segment(ep.number, t0, t1, sc["_beats"])], hook_beat=sc["_hb"], cliff_beat=sc["_cb"])
        p.scores = {k: round(v, 3) for k, v in sc.items() if not k.startswith("_")}
        p.notes = sc["_notes"] + ["proposed by LLM"] + ([repaired] if repaired else [])
        # if code changed the cut, the model's hook/cliff text may no longer match: keep it only when unchanged
        p.llm = {"proposal": {"hook": "" if repaired else str(it.get("hook", ""))[:300],
                              "cliffhanger": "" if repaired else str(it.get("cliffhanger", ""))[:300],
                              "why": str(it.get("why", ""))[:300]}}
        out.append(p)
    return out, log


def _repair_length(ep: Episode, t0: float, t1: float, sp: list[float], epts: list[float]):
    """The model often picks just the peak (10-19 s) or a whole scene (60 s+).  Try the
    smallest legal fix in both directions and keep whichever the rules score higher:
      too short -> extend the end forward, or start earlier (more build-up)
      too long  -> pull the end in, or start later
    Returns (start, end, how) or None."""
    opts = []
    if t1 - t0 < MIN_LEN:
        e = [x for x in epts if t0 + MIN_LEN <= x <= t0 + MAX_LEN]
        if e:
            opts.append((t0, min(e), "extended the end"))
        s = [x for x in sp if t1 - MAX_LEN <= x <= t1 - MIN_LEN]
        if s:
            opts.append((max(s), t1, "started earlier"))
    else:
        e = [x for x in epts if t0 + MIN_LEN <= x <= t0 + MAX_LEN]
        if e:
            opts.append((t0, max(e), "pulled the end in"))
        s = [x for x in sp if t1 - MAX_LEN <= x <= t1 - MIN_LEN]
        if s:
            opts.append((min(s), t1, "started later"))
    best = None
    for a, b, how in opts:
        sc = score_window(ep, a, b)
        if sc["_hb"] is None:
            continue
        if best is None or sc["rules"] > best[0]:
            best = (sc["rules"], a, b, how)
    return None if best is None else best[1:]
