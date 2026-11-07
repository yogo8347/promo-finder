"""Check that the AI's hook / cliffhanger text describes what is actually IN the clip.

Found by reading the final report: for the #1 post the judge wrote that the stranger "has knocked down a
guard", but the clip cuts on "I said... let her go." and the punch comes after the cut. Timestamps were
always checked by code; the AI's descriptions were not. Two layers now:

1. code:  every quote inside the AI's text must appear in the clip (fuzzy, so small paraphrases pass)
2. AI fact-check of the top N, with the transcript in front of it, batched. A corrected text goes through
   check 1 again; if it still fails, code falls back to the real line from the script.

Every correction is recorded (p.llm["text_check"]) and counted in the QA report.
"""
from __future__ import annotations

import re

from .candidates import Promo
from .judge import EDITOR_SYSTEM
from .llm import LLM, LLMError

_STOP = set("the a an and or of to in on at for with by is are was were be her his him she he they it its this that as from".split())


def _norm_words(t: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", t.lower()) if w not in _STOP]


def quotes(text: str) -> list[str]:
    """Quoted phrases of 3+ words ("..." or '...', not apostrophes)."""
    qs = re.findall(r'["“]([^"”]{4,})["”]', text) + re.findall(r"(?:^|[\s(:])'([^']{4,}?)'(?=[\s.,;:!?)]|$)", text)
    return [q for q in qs if len(_norm_words(q)) >= 3]


def clip_text(p: Promo) -> str:
    return " ".join(b.text for b in p.beats() if b.kind != "camera")


def unmatched_quotes(text: str, p: Promo, min_overlap: float = 0.7) -> list[str]:
    have = set(_norm_words(clip_text(p)))
    out = []
    for q in quotes(text):
        w = _norm_words(q)
        if sum(x in have for x in w) / len(w) < min_overlap:
            out.append(q)
    return out


def last_line(p: Promo) -> str:
    for b in reversed(p.segments[-1].beats):
        if b.kind == "dialogue":
            name = " ".join(x[:1].upper() + x[1:].lower() for x in b.speaker.split())
            return f'{name}: "{b.text}"'
    for b in reversed(p.segments[-1].beats):
        if b.kind != "camera":
            return b.text
    return ""


def first_line(p: Promo) -> str:
    seg = p.segments[0]
    for b in seg.beats:
        if b.kind != "camera" and b.start <= seg.start + 3.0:
            if b.kind == "dialogue":
                name = " ".join(x[:1].upper() + x[1:].lower() for x in b.speaker.split())
                return f'{name}: "{b.text}"'
            return b.text
    return ""


VERIFY_PROMPT = """Fact-check short descriptions of promo clips against the clips' transcripts.
For EACH clip, the HOOK must describe something in the first 3 seconds, and the CUT must describe what is on
screen in the last few seconds. A description is WRONG if it mentions an event, line or outcome that does not
appear in that clip's transcript (for example something that only happens after the clip ends).
If wrong, write a fixed version (max 20 words) that quotes or describes the actual line.

{clips}

Answer JSON only: {{"clips": [{{"clip": "C1", "hook_ok": true/false, "hook_fixed": "...", "cut_ok": true/false, "cut_fixed": "..."}}, ...]}}
(leave a *_fixed field empty when that part is ok)"""


def _transcript(p: Promo) -> str:
    rows, off = [], 0.0
    for s in p.segments:
        for b in s.beats:
            if b.kind == "camera":
                continue
            rows.append(f"[{off + b.start - s.start:5.1f}s] {b.label()}")
        off += s.dur
    return "\n".join(rows)


def verify(llm: LLM, promos: list[Promo], batch: int = 5) -> dict:
    """Fact-check hook/cliffhanger text of `promos` in place. Returns counts for the QA report."""
    counts = {"checked": 0, "hook_fixed": 0, "cut_fixed": 0, "fallback_to_script": 0, "failed_calls": 0}
    for i in range(0, len(promos), batch):
        group = promos[i:i + batch]
        blocks = []
        for k, p in enumerate(group, 1):
            h = p.llm.get("hook") or p.llm.get("proposal", {}).get("hook", "")
            c = p.llm.get("cliffhanger") or p.llm.get("proposal", {}).get("cliffhanger", "")
            blocks.append(f"=== C{k} ({p.duration:.0f}s) ===\n{_transcript(p)}\nHOOK: {h}\nCUT: {c}")
        try:
            r = llm.complete_json(VERIFY_PROMPT.format(clips="\n\n".join(blocks)), EDITOR_SYSTEM)
            rows = r.get("clips", r) if isinstance(r, dict) else r
            by = {str(x.get("clip", "")).upper(): x for x in rows if isinstance(x, dict)}
        except (LLMError, AttributeError, TypeError):
            counts["failed_calls"] += 1
            by = {}
        for k, p in enumerate(group, 1):
            x = by.get(f"C{k}")
            # code layer runs regardless of whether the AI check answered
            note = {}
            for part, key, okk, fixk, fallback in (("hook", "hook", "hook_ok", "hook_fixed", first_line),
                                                    ("cut", "cliffhanger", "cut_ok", "cut_fixed", last_line)):
                cur = p.llm.get(key) or p.llm.get("proposal", {}).get(key, "")
                if not cur:
                    continue
                bad_q = unmatched_quotes(cur, p)
                ai_bad = x is not None and x.get(okk) is False
                if not (bad_q or ai_bad):
                    continue
                fixed = str((x or {}).get(fixk, "")).strip()
                if not fixed or unmatched_quotes(fixed, p):
                    fixed = fallback(p)
                    counts["fallback_to_script"] += 1
                p.llm[f"{key}_ai_original"] = cur
                p.llm[key] = fixed
                counts[f"{part}_fixed"] += 1
                note[part] = {"was": cur, "now": fixed, "why": "quote not in clip" if bad_q else "AI fact-check"}
            if x is not None or note:
                counts["checked"] += 1
            if note:
                p.llm["text_check"] = note
    return counts
