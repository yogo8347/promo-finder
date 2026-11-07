"""A second, independent reference set: a DIFFERENT model picks the promo moments, blind.

The hand-made reference picks were chosen by the tool's builder, after seeing an early ranking, so recall
against them is optimistic. This script asks a different model (default: Gemini 3.8 Flash; the tool itself
runs on Gemma 4) to read every episode and pick the moments it would post. It never sees the tool's
candidates or scores. Its picks are written in the reference-picks format, so `qa.recall_report` can score
any run against them.

    python eval/independent_picks.py --input data/input --out eval/independent_picks.json
    python eval/independent_picks.py --score outputs/02_plus_gemma4_26b/promos.json

The output quotes the scripts, so it is git-ignored like the hand-made picks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from promo_finder.llm import LLM  # noqa: E402
from promo_finder.parser import fmt, load_show, to_seconds  # noqa: E402

PROMPT = """You are a performance-marketing editor for a vertical microdrama app (ReelShort / DramaBox style).
Below are {n} episodes of one show, line by line (ID, start time, line). Pick the {k} BEST promo moments in these
episodes: clips you would put ad budget behind on TikTok / Reels / Shorts for viewers who have never seen the show.
A great promo is one continuous 20-60 second stretch of ONE episode that opens on a hook in its first 3 seconds
(a slap, a humiliation, a reveal, a shocking line), cuts on an unresolved cliffhanger, and makes sense to a stranger.

{episodes}

Answer JSON only: {{"picks": [{{"episode": <number>, "start_line": "<ID>", "end_line": "<ID>",
"moment": "<one line: what happens>", "post_today": true/false}}, ...]}}"""


def episode_block(ep) -> tuple[str, dict]:
    rows, ids = [f"### Episode {ep.number}: {ep.logline}"], {}
    k = 0
    for b in ep.beats:
        if b.kind == "camera":
            continue
        k += 1
        i = f"E{ep.number}-L{k}"
        ids[i] = b
        rows.append(f"{i} [{fmt(b.start)}] {b.label()}")
    return "\n".join(rows), ids


def make_picks(input_dir: str, model: str, per_batch: int, batch: int) -> dict:
    show = load_show(input_dir)
    llm = LLM("gemini", model)
    llm.fallbacks = ["gemini-3.7-flash", "gemini-3.6-flash"]
    picks = []
    for i in range(0, len(show), batch):
        eps = show[i:i + batch]
        blocks, ids = [], {}
        for e in eps:
            t, m = episode_block(e)
            blocks.append(t)
            ids.update(m)
        r = llm.complete_json(PROMPT.format(n=len(eps), k=per_batch, episodes="\n\n".join(blocks)))
        rows = r.get("picks", []) if isinstance(r, dict) else r
        for x in rows[:per_batch]:
            a, b = ids.get(str(x.get("start_line"))), ids.get(str(x.get("end_line")))
            if not a or not b or a.episode != b.episode or b.end <= a.start:
                print(f"skipped an unusable pick: {x}", file=sys.stderr)
                continue
            picks.append({"id": f"I{len(picks) + 1:02d}", "episode": a.episode, "start": fmt(a.start), "end": fmt(b.end),
                          "moment": str(x.get("moment", ""))[:200], "tier": "A" if x.get("post_today") else "B"})
        print(f"episodes {eps[0].number}-{eps[-1].number}: {len(picks)} picks so far ({llm.model})", file=sys.stderr)
    first = show[0]
    return {"_about": f"Independent reference picks made blind by {llm.model} from the full scripts. It never saw the tool's "
                      "candidates or scores. One model's taste: a second opinion, not ground truth.",
            "show": first.logline[:80], "fingerprint": first.logline[:40], "picks": picks}


def score(promos_json: str, picks_json: str) -> None:
    refs = json.loads(Path(picks_json).read_text(encoding="utf-8"))["picks"]
    promos = json.loads(Path(promos_json).read_text(encoding="utf-8"))["promos"]

    def hit(p, r):
        r0, r1 = to_seconds(r["start"]), to_seconds(r["end"])
        return any(s["episode"] == r["episode"] and
                   max(0, min(s["end_s"], r1) - max(s["start_s"], r0)) / max(1e-6, min(s["end_s"] - s["start_s"], r1 - r0)) >= 0.5
                   for s in p["segments"])
    nA = sum(r["tier"] == "A" for r in refs)
    for k in (10, 20, 30, len(promos)):
        top = promos[:k]
        f = sum(any(hit(p, r) for p in top) for r in refs)
        fa = sum(r["tier"] == "A" and any(hit(p, r) for p in top) for r in refs)
        print(f"top {k if k != len(promos) else 'all'}: {f}/{len(refs)} independent picks found (post-today: {fa}/{nA})")
    for r in refs:
        rk = next((p["rank"] for p in promos if hit(p, r)), None)
        print(f"  {r['id']} E{r['episode']} {r['start']}-{r['end']} [{r['tier']}] rank {rk}: {r['moment'][:70]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/input")
    ap.add_argument("--out", default="eval/independent_picks.json")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--per-batch", type=int, default=3)
    ap.add_argument("--batch", type=int, default=10, help="episodes per call")
    ap.add_argument("--score", default=None, help="promos.json to score against --out")
    a = ap.parse_args()
    if a.score:
        score(a.score, a.out)
    else:
        data = make_picks(a.input, a.model, a.per_batch, a.batch)
        Path(a.out).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {len(data['picks'])} picks to {a.out}")
