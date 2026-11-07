"""Bring in another tool's promo slate so both pools compete on equal terms.

Accepted formats (auto-detected per item):
  * this tool's promos.json:  {"promos": [{"segments": [{"episode", "start_s", "end_s"}, ...]}]}
  * a flat slate:             {"promos": [{"episode", "start_sec", "end_sec"}]}   (or a bare list)

Only the episode and the times are used. The other tool's hook / cliffhanger / scores are ignored, so
the imported clips are re-judged blind with the same prompts as ours. Their cut points are snapped to
this tool's legal cut points (shot starts, line ends), so an imported clip obeys the same rules.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import candidates as C
from . import features as F
from .parser import Episode


def _items(data) -> list[list[tuple[int, float, float]]]:
    rows = data.get("promos", data) if isinstance(data, dict) else data
    out = []
    for p in rows:
        if "segments" in p:
            out.append([(int(s["episode"]), float(s.get("start_s", s.get("start_sec"))),
                         float(s.get("end_s", s.get("end_sec")))) for s in p["segments"]])
        elif "episode" in p:
            out.append([(int(p["episode"]), float(p.get("start_sec", p.get("start_s"))),
                         float(p.get("end_sec", p.get("end_s"))))])
    return out


def _snap(ep: Episode, t0: float, t1: float) -> tuple[float, float] | None:
    sp = [t for t in C.start_points(ep) if t <= t0 + 1.5] or C.start_points(ep)[:1]
    epts = [t for t in C.end_points(ep) if t >= t1 - 1.5] or C.end_points(ep)[-1:]
    if not sp or not epts:
        return None
    s = max(sp)
    e = min(epts, key=lambda t: abs(t - t1))
    return (s, e) if e > s else None


def load_external(path: str | Path, eps: dict[int, Episode], tag: str = "external") -> tuple[list[C.Promo], list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    promos, log = [], []
    for n, segs in enumerate(_items(data), 1):
        built = []
        for e, t0, t1 in segs:
            ep = eps.get(e)
            snapped = _snap(ep, t0, t1) if ep else None
            if not snapped:
                break
            built.append(C.Segment(e, snapped[0], snapped[1], ep.window(*snapped)))
        if len(built) != len(segs):
            log.append(f"{tag} #{n}: could not map onto the scripts, skipped")
            continue
        dur = sum(s.dur for s in built)
        if not (C.MIN_LEN <= dur <= C.MAX_LEN):
            log.append(f"{tag} #{n}: {dur:.1f}s after snapping to legal cuts, outside 20-60s, skipped")
            continue
        if len(built) == 1:
            seg = built[0]
            sc = C.score_window(eps[seg.episode], seg.start, seg.end)
            p = None
            if sc["_hb"] is not None:
                p = C.Promo("single", [C.Segment(seg.episode, seg.start, seg.end, sc.pop("_beats"))],
                            hook_beat=sc.pop("_hb"), cliff_beat=sc.pop("_cb"))
                p.notes = sc.pop("_notes")
                p.scores = {k: round(v, 3) for k, v in sc.items()}
        else:
            p = C.Promo("montage", built)
            first, last = built[0], built[-1]
            p.hook_beat = max((b for b in first.beats if b.start <= first.start + 3.0 and b.kind != "camera"),
                              key=F.beat_intensity, default=None)
            p.cliff_beat = next((b for b in reversed(last.beats) if b.kind != "camera"), None)
            p.scores = {"rules": 0.5}
        if p is None:
            log.append(f"{tag} #{n}: failed the cut rules, skipped")
            continue
        p.notes.append(f"from {tag} slate (#{n})")
        p.scores["source_rank"] = n
        promos.append(p)
    return promos, log
