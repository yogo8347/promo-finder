"""Write every run to outputs/<run>/ : JSON (machine), CSV (sheet), Markdown (editor)."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from . import rank
from .parser import fmt


def promo_dict(p, rank_no: int) -> dict:
    f = rank.fields(p)
    return {
        "rank": rank_no, "id": p.id, "type": p.kind,
        "total_duration_s": round(p.duration, 2),
        "segments": [s.to_dict() for s in p.segments],
        "hook": f["hook"], "hook_at_s": f["hook_at_s"], "cliffhanger": f["cliffhanger"],
        "why_it_works": f["why"], "text_source": f["source"],
        "scores": p.scores, "llm": {k: v for k, v in p.llm.items() if k != "proposal"} or None,
        "llm_proposal": p.llm.get("proposal"),
        "flags": rank.flags(p), "notes": p.notes,
        "transcript": p.transcript(),
    }


def write_all(res, cfg) -> Path:
    name = cfg.run_name or time.strftime("run_%Y%m%d_%H%M%S") + "_" + rank.slug(res.meta["engine"], 30)
    out = Path(cfg.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    ranked = res.ranked

    # 1) everything, machine-readable
    (out / "promos.json").write_text(json.dumps(
        {"meta": res.meta, "promos": [promo_dict(p, i + 1) for i, p in enumerate(ranked)]},
        indent=2, ensure_ascii=False), encoding="utf-8")

    # 2) flat sheet for an editor / Google Sheets
    with open(out / "promos.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "id", "type", "duration_s", "segments", "hook", "hook_at_s", "cliffhanger",
                    "why_it_works", "final", "rules", "hook_r", "cliff_r", "standalone_r",
                    "llm_hook", "llm_cliff", "llm_standalone", "llm_verdict", "flags"])
        for i, p in enumerate(ranked):
            f = rank.fields(p)
            j = p.llm or {}
            w.writerow([i + 1, p.id, p.kind, round(p.duration, 1), rank.segment_str(p), f["hook"], f["hook_at_s"],
                        f["cliffhanger"], f["why"], p.scores.get("final"), p.scores.get("rules"),
                        p.scores.get("hook"), p.scores.get("cliff"), p.scores.get("standalone"),
                        j.get("hook_score", ""), j.get("cliff_score", ""), j.get("standalone_score", ""),
                        j.get("verdict", ""), " | ".join(rank.flags(p))])

    # 3) the editor's report
    md = [f"# Promo picks: {Path(cfg.input_dir).name}", "",
          f"Engine: **{res.meta['engine']}**  |  {res.meta['episodes']} episodes, {res.meta['show_minutes']} min, "
          f"{res.meta['beats']} script beats  |  {res.meta['candidates']} candidates -> {len(ranked)} after de-dup  |  "
          f"{res.meta['seconds']}s run, {res.meta['llm_calls']} LLM calls", "",
          "Timestamps are episode-relative (MM:SS.ss), taken straight from the script. "
          "`hook @` = seconds into the promo where the hook lands.", "",
          "## Posting order (what to post first)", "",
          "| # | promo | segments | dur | why first |", "|---|---|---|---|---|"]
    for n, (p, why) in enumerate(res.plan, 1):
        md.append(f"| {n} | {p.id} | {rank.segment_str(p)} | {p.duration:.0f}s | {why} |")
    md += ["", f"## Top {min(cfg.top_n, len(ranked))} promos by score", ""]
    for i, p in enumerate(ranked[:cfg.top_n]):
        f = rank.fields(p)
        j = p.llm or {}
        md += [f"### {i + 1}. {p.id} ({p.kind}, {p.duration:.1f}s), score {p.scores['final']:.2f}", ""]
        for k, s in enumerate(p.segments, 1):
            md.append(f"- Segment {k}: **Episode {s.episode}, {fmt(s.start)} to {fmt(s.end)}** ({s.dur:.1f}s)")
        md += [f"- **Hook** (@{f['hook_at_s']}s): {f['hook']}",
               f"- **Cliffhanger:** {f['cliffhanger']}",
               f"- **Why it works:** {f['why']}"]
        sc = p.scores
        line = (f"- Scores: rules {sc.get('rules', 0):.2f} (hook {sc.get('hook', 0):.2f}, cliff {sc.get('cliff', 0):.2f}, "
                f"standalone {sc.get('standalone', 0):.2f})")
        if "hook_score" in j:
            line += (f"; LLM hook {j['hook_score']:.0f}/10, cliff {j['cliff_score']:.0f}/10, "
                     f"standalone {j['standalone_score']:.0f}/10, verdict **{j['verdict']}**")
        md.append(line)
        fl = rank.flags(p)
        if fl:
            md.append("- Flags: " + "; ".join(fl))
        md += ["", "<details><summary>transcript</summary>", "", "```", p.transcript(), "```", "</details>", ""]
    (out / "promos.md").write_text("\n".join(md), encoding="utf-8")

    # 4) QA + run metadata + LLM log
    q = res.qa
    qa_md = ["# Quality report", "", f"Engine: {res.meta['engine']}", "",
             "## 1. Hard validators (every promo)", "",
             f"Checked {q['n_checked']} promos for: 20-60s length, hook inside the first 3s, no cut inside a line "
             f"of dialogue, segments in story order, timestamps inside the episode.", "",
             f"**Failures: {len(q['validator_failures'])}**", ""]
    for pid, errs in q["validator_failures"].items():
        qa_md.append(f"- {pid}: " + "; ".join(errs))
    r = q.get("recall") or {}
    if r.get("skipped"):
        qa_md += ["", "## 2. Recall against reference picks", "", f"Skipped: {r['skipped']}. "
                  "Add an `eval/reference_picks.json` for this show to measure recall."]
    elif r:
        qa_md += ["", "## 2. Recall against hand-picked reference promos", "",
                  f"{r['n_reference']} reference moments in `eval/reference_picks.json` (tier A = would post today).", "",
                  "| cut-off | all refs found | tier A found |", "|---|---|---|"]
        for k, v in r["recall"].items():
            qa_md.append(f"| {k} | {v['all_refs']} | {v['tier_A']} |")
        qa_md += ["", f"Share of the top 20 that match a reference pick: {r['top20_matching_a_reference']} "
                  "(the reference list is incomplete, so a non-match is not necessarily a bad pick).", "",
                  "| ref | ep | tier | moment | tool's rank |", "|---|---|---|---|---|"]
        for x in r["per_pick"]:
            qa_md.append(f"| {x['id']} | {x['episode']} | {x['tier']} | {x['moment']} | "
                         f"{x['found_at_rank'] if x['found_at_rank'] else '**missed**'} |")
    t = q.get("tournament") or {}
    if t and "skipped" not in t:
        qa_md += ["", "## 3. Head-to-head tournament (re-ranks the top of the list)", "",
                  f"{t['games']} blind A-vs-B games over {t['rounds']} Swiss rounds, pool of {t['pool']} clips "
                  f"({t['failed_calls']} failed calls). {t.get('top10_changed', 0)} of the top 10 positions changed.", "",
                  "| judge check | result | what good looks like |", "|---|---|---|",
                  f"| clip shown first (A) wins | {t['position_bias_A_win_rate']:.0%} | about 50% |",
                  f"| same verdict when A and B are swapped | {t['swap_recheck']} | high; low = coin flips |"]
    src = q.get("sources") or {}
    if any(len(v) > 1 or "ours" not in v for v in src.values()):
        qa_md += ["", "## 4. Which slate found the moments (when other slates were merged in)", "",
                  "| cut-off | " + " | ".join(sorted({k for v in src.values() for k in v})) + " |",
                  "|---|" + "---|" * len({k for v in src.values() for k in v})]
        keys = sorted({k for v in src.values() for k in v})
        for cut, v in src.items():
            qa_md.append(f"| {cut} | " + " | ".join(str(v.get(k, 0)) for k in keys) + " |")
    qa_md += ["", "## Flags on the top 20 (look at these first)", ""]
    for p in ranked[:20]:
        fl = rank.flags(p)
        if fl:
            qa_md.append(f"- {p.id} {rank.segment_str(p)}: " + "; ".join(fl))
    (out / "qa_report.md").write_text("\n".join(qa_md), encoding="utf-8")
    # transcript-free copy (hook / cliffhanger / why can still quote a short line: share privately)
    public = [{k: v for k, v in promo_dict(p, i + 1).items() if k not in ("transcript", "llm", "llm_proposal", "notes")}
              for i, p in enumerate(ranked)]
    (out / "promos.public.json").write_text(json.dumps({
        "_notice": "Timestamps, scores and flags without transcripts. Hook / cliffhanger / why can still quote a "
                   "short line of the confidential scripts, so share this by private link too.",
        "promos": public}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps({**res.meta, "qa": q}, indent=2, default=str), encoding="utf-8")
    (out / "llm_log.txt").write_text("\n".join(res.log) or "(no LLM events)", encoding="utf-8")
    return out
