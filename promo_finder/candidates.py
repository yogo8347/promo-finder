"""Turn parsed episodes into promo candidates.

* single  : one continuous 20-60 s stretch of one episode
* montage : 2-4 short moments, in story order, from different episodes, 30-60 s total

Every start/end is snapped to a real cut point in the script (a camera-shot boundary or
the edge of a line), and never lands in the middle of a line of dialogue.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from . import features as F
from .parser import Beat, Episode, fmt

MIN_LEN, MAX_LEN = 20.0, 60.0


@dataclass
class Segment:
    episode: int
    start: float
    end: float
    beats: list[Beat] = field(default_factory=list, repr=False)

    @property
    def dur(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {"episode": self.episode, "start": fmt(self.start), "end": fmt(self.end),
                "start_s": round(self.start, 2), "end_s": round(self.end, 2),
                "duration_s": round(self.dur, 2)}


@dataclass
class Promo:
    kind: str                       # single | montage
    segments: list[Segment]
    hook_beat: Beat | None = None
    cliff_beat: Beat | None = None
    scores: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    id: str = ""
    # filled by the LLM judge (optional)
    llm: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return sum(s.dur for s in self.segments)

    @property
    def episodes(self) -> list[int]:
        return [s.episode for s in self.segments]

    def beats(self) -> list[Beat]:
        return [b for s in self.segments for b in s.beats]

    def transcript(self, with_camera: bool = False) -> str:
        """Promo-relative time on the left (what the viewer sees), source time in brackets."""
        out, offset = [], 0.0
        for n, s in enumerate(self.segments):
            if len(self.segments) > 1:
                out.append(f"--- segment {n + 1}: Episode {s.episode} {fmt(s.start)}-{fmt(s.end)} ---")
            for b in s.beats:
                if b.kind == "camera" and not with_camera:
                    continue
                rel = offset + (b.start - s.start)
                out.append(f"{fmt(rel)} [E{b.episode:02d} {fmt(b.start)}] {b.label()}")
            offset += s.dur
        return "\n".join(out)


# ------------------------------------------------------------------ cut points ----------
def _lines_straddling(ep: Episode, t: float, tol_before: float = 0.25, tol_after: float = 0.25) -> list[Beat]:
    return [b for b in ep.beats if b.kind == "dialogue" and b.start < t - tol_before and b.end > t + tol_after]


def start_points(ep: Episode) -> list[float]:
    """Clean in-points: the start of a camera shot that opens on (or just before) content."""
    pts = set()
    cams = [b for b in ep.beats if b.kind == "camera"]
    for b in ep.beats:
        if b.kind == "camera":
            continue
        t = b.start
        cover = [c for c in cams if c.start <= t + 1e-6 and t - c.start <= 1.2]
        if cover:
            t = max(c.start for c in cover)
        if not _lines_straddling(ep, t):
            pts.add(round(t, 2))
    return sorted(pts)


def end_points(ep: Episode) -> list[float]:
    """Clean out-points: just after a line/action ends, held to the end of the shot if close."""
    pts = set()
    cams = [b for b in ep.beats if b.kind == "camera"]
    for b in ep.beats:
        if b.kind == "camera":
            continue
        t = b.end
        hold = [c.end for c in cams if c.start <= t <= c.end and c.end - t <= 1.5]
        if hold:
            t = max(hold)
        t = min(t, ep.duration)
        strad = _lines_straddling(ep, t)
        if strad:
            continue
        pts.add(round(t, 2))
    pts.add(round(ep.duration, 2))
    return sorted(pts)


# ------------------------------------------------------------------ scoring -------------
WEIGHTS = {"hook": 0.32, "cliff": 0.30, "standalone": 0.18, "density": 0.12, "duration": 0.08}


def score_window(ep: Episode, t0: float, t1: float) -> dict:
    inc = ep.window(t0, t1)
    h, hb, hn = F.hook_strength(inc, t0)
    c, cb, cn = F.cliff_strength(ep.beats, inc, t1, is_episode_end=abs(t1 - ep.duration) < 2.0)
    s, sn = F.standalone_strength(inc)
    d = F.density(inc, t0, t1)
    fit = F.duration_fit(t1 - t0)
    total = (WEIGHTS["hook"] * h + WEIGHTS["cliff"] * c + WEIGHTS["standalone"] * s
             + WEIGHTS["density"] * d + WEIGHTS["duration"] * fit)
    return {"hook": h, "cliff": c, "standalone": s, "density": d, "duration": fit,
            "rules": total, "_hb": hb, "_cb": cb, "_notes": hn + cn + sn, "_beats": inc}


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def single_candidates(ep: Episode, per_episode: int = 4, max_starts: int = 30) -> list[Promo]:
    starts = start_points(ep)
    ends = end_points(ep)
    # pre-filter in-points by hook strength (cheap) to keep the search small
    ranked = []
    for t0 in starts:
        h, hb, _ = F.hook_strength(ep.window(t0, t0 + 3.01), t0)
        if hb is not None and h >= 0.3:
            ranked.append((h, t0))
    ranked.sort(reverse=True)
    pool = []
    for _, t0 in ranked[:max_starts]:
        for t1 in ends:
            if not (MIN_LEN <= t1 - t0 <= MAX_LEN):
                continue
            sc = score_window(ep, t0, t1)
            pool.append((sc["rules"], t0, t1, sc))
    pool.sort(key=lambda x: -x[0])
    kept: list[Promo] = []
    for total, t0, t1, sc in pool:
        if any(_iou((t0, t1), (k.segments[0].start, k.segments[0].end)) > 0.35 for k in kept):
            continue
        p = Promo("single", [Segment(ep.number, t0, t1, sc.pop("_beats"))],
                  hook_beat=sc.pop("_hb"), cliff_beat=sc.pop("_cb"))
        p.notes = sc.pop("_notes")
        p.scores = {k: round(v, 3) for k, v in sc.items()}
        kept.append(p)
        if len(kept) >= per_episode:
            break
    return kept


# ------------------------------------------------------------------ montages ------------
@dataclass
class Moment:
    ep: Episode
    seg: Segment
    hook: float
    cliff: float
    standalone: float
    density: float
    hook_beat: Beat | None
    cliff_beat: Beat | None


def moments(ep: Episode, lo: float = 5.0, hi: float = 18.0, per_episode: int = 3) -> list[Moment]:
    """Short, punchy beats that could open, build or close a montage."""
    starts, ends = start_points(ep), end_points(ep)
    pool = []
    for t0 in starts:
        h, hb, _ = F.hook_strength(ep.window(t0, t0 + 3.01), t0)
        if hb is None or h < 0.45:
            continue
        for t1 in ends:
            if not (lo <= t1 - t0 <= hi):
                continue
            inc = ep.window(t0, t1)
            c, cb, _ = F.cliff_strength(ep.beats, inc, t1, abs(t1 - ep.duration) < 2)
            s, _ = F.standalone_strength(inc)
            d = F.density(inc, t0, t1)
            pool.append((0.4 * h + 0.3 * c + 0.15 * s + 0.15 * d, Moment(ep, Segment(ep.number, t0, t1, inc), h, c, s, d, hb, cb)))
    pool.sort(key=lambda x: -x[0])
    out: list[Moment] = []
    for _, m in pool:
        if any(_iou((m.seg.start, m.seg.end), (k.seg.start, k.seg.end)) > 0.2 for k in out):
            continue
        out.append(m)
        if len(out) >= per_episode:
            break
    return out


def montage_candidates(eps: list[Episode], n: int = 12, top_moments: int = 45) -> list[Promo]:
    allm = [m for e in eps for m in moments(e)]
    if len(allm) < 2:
        return []
    openers = sorted(allm, key=lambda m: -(m.hook + 0.3 * m.standalone))[:12]
    closers = sorted(allm, key=lambda m: -(m.cliff + 0.2 * m.hook))[:15]
    fillers = sorted(allm, key=lambda m: -(m.density + m.hook + m.standalone))[:top_moments]
    out: list[tuple[float, list[Moment]]] = []
    for o, c in itertools.product(openers, closers):
        if c.ep.number <= o.ep.number:
            continue
        base = o.seg.dur + c.seg.dur
        if base > MAX_LEN:
            continue
        best_combo, best_val = [o, c], -1.0
        mids = [m for m in fillers if o.ep.number < m.ep.number < c.ep.number]
        options = [[]] + [[m] for m in mids] + [list(x) for x in itertools.combinations(mids, 2)]
        for mid in options:
            combo = [o] + sorted(mid, key=lambda m: (m.ep.number, m.seg.start)) + [c]
            dur = sum(m.seg.dur for m in combo)
            if not (30.0 <= dur <= MAX_LEN):
                continue
            joins = len(combo) - 1
            # same weights as single clips so both compete on one scale; jumping across
            # episodes makes a clip harder to follow and every join costs editor time
            stand = sum(m.standalone for m in combo) / len(combo) - 0.06 * joins
            val = (WEIGHTS["hook"] * o.hook + WEIGHTS["cliff"] * c.cliff
                   + WEIGHTS["standalone"] * stand
                   + WEIGHTS["density"] * sum(m.density for m in combo) / len(combo)
                   + WEIGHTS["duration"] * F.duration_fit(dur)
                   - 0.03 * joins)
            if val > best_val:
                best_val, best_combo = val, combo
        if best_val > 0:
            out.append((best_val, best_combo))
    out.sort(key=lambda x: -x[0])
    promos: list[Promo] = []
    used: list[set] = []
    uses: dict = {}
    for val, combo in out:
        key = {(m.ep.number, round(m.seg.start)) for m in combo}
        if any(len(key & u) >= 2 for u in used):     # avoid near-duplicate montages
            continue
        if any(uses.get(k, 0) >= 2 for k in key):    # one moment may open/close at most 2 montages
            continue
        used.append(key)
        for k in key:
            uses[k] = uses.get(k, 0) + 1
        p = Promo("montage", [m.seg for m in combo], hook_beat=combo[0].hook_beat, cliff_beat=combo[-1].cliff_beat)
        p.scores = {"hook": round(combo[0].hook, 3), "cliff": round(combo[-1].cliff, 3),
                    "standalone": round(sum(m.standalone for m in combo) / len(combo) - 0.06 * (len(combo) - 1), 3),
                    "density": round(sum(m.density for m in combo) / len(combo), 3),
                    "duration": round(F.duration_fit(p.duration), 3), "rules": round(val, 3)}
        p.notes = [f"montage of {len(combo)} moments; needs {len(combo) - 1} edit join(s)"]
        promos.append(p)
        if len(promos) >= n:
            break
    return promos
