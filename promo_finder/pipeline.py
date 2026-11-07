"""End-to-end run:  scripts folder -> ranked promos + reports.

    parse (code) -> candidates (rules) -> [LLM propose] -> [LLM judge top K] -> rank -> validate -> write
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import candidates as C
from . import judge as J
from . import qa, rank
from . import tournament as T
from .external import load_external
from .llm import LLM
from .parser import load_show


@dataclass
class RunConfig:
    input_dir: str = "data/input"
    out_dir: str = "outputs"
    provider: str | None = None
    model: str | None = None
    per_episode: int = 4          # single-clip candidates kept per episode
    montages: int = 12
    judge_top: int = 60           # how many candidates the LLM judges (cost control)
    judge_batch: int = 3          # clips per judge call (fewer calls on free tiers)
    llm_propose: bool = True      # let the LLM read whole episodes and suggest clips too
    top_n: int = 30               # promos in the final report
    reference: str = "eval/reference_picks.json"
    run_name: str = ""
    merge: list = field(default_factory=list)   # other tools' slates (json) to import and compete
    tournament: int = 28          # head-to-head pool size (0 = off); needs an LLM
    rounds: int = 5               # Swiss rounds
    recheck: int = 10             # games replayed with A/B swapped, to measure the judge


@dataclass
class RunResult:
    ranked: list = field(default_factory=list)
    plan: list = field(default_factory=list)
    episodes: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    qa: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
    out_path: Path | None = None
    qa_tournament: dict = field(default_factory=dict)


def _source(p) -> str:
    for n in p.notes:
        if n.startswith("from ") and n.endswith(")") and " slate " in n:
            return n[5:n.index(" slate ")]
    return "ours"


def _source_share(ranked) -> dict:
    out = {}
    for k in (10, 20, 30):
        c: dict[str, int] = {}
        for p in ranked[:k]:
            f = next((n[10:] for n in p.notes if n.startswith("found by: ")), "ours")
            c[f] = c.get(f, 0) + 1
        out[f"top{k}"] = c
    return out


def run(cfg: RunConfig, progress: Callable[[str, float], None] | None = None) -> RunResult:
    say = progress or (lambda msg, frac: print(f"[{frac:4.0%}] {msg}"))
    t_start = time.time()
    res = RunResult()
    llm = LLM(cfg.provider, cfg.model)
    if llm.enabled:   # fail fast with a clear message instead of 100 identical errors
        try:
            llm.complete('Reply with JSON {"ok": true}', want_json=True)
        except Exception as e:
            res.log.append(f"LLM pre-flight failed, falling back to rules only: {e}")
            say(f"LLM unavailable ({str(e)[:120]}); continuing with rules only", 0.0)
            llm = LLM("none")

    say("parsing scripts", 0.0)
    show = load_show(cfg.input_dir)
    if not show:
        raise SystemExit(f"no screenplay .txt files found in {cfg.input_dir}")
    eps = {e.number: e for e in show}
    res.episodes = eps

    say(f"finding single-clip candidates in {len(show)} episodes", 0.05)
    cands: list[C.Promo] = []
    for i, e in enumerate(show):
        cands += C.single_candidates(e, per_episode=cfg.per_episode)
        say(f"episode {e.number}: {len(cands)} candidates so far", 0.05 + 0.35 * (i + 1) / len(show))
    say("building cross-episode montages", 0.42)
    cands += C.montage_candidates(show, n=cfg.montages)
    for path in cfg.merge:
        tag = Path(path).stem
        ext, log = load_external(path, eps, tag=tag)
        cands += ext
        res.log += log
        say(f"imported {len(ext)} clips from {path}", 0.44)

    if llm.enabled and cfg.llm_propose:
        for i, e in enumerate(show):
            props, log = J.propose(llm, e)
            cands += props
            res.log += log
            say(f"LLM read episode {e.number} ({len(props)} proposals)", 0.45 + 0.2 * (i + 1) / len(show))

    for p in cands:
        p.scores["final"] = rank.final_score(p)

    if llm.enabled:
        # judge the best rule candidates + every LLM proposal + every montage
        by_rules = sorted(cands, key=lambda p: -p.scores["rules"])
        to_judge = by_rules[:cfg.judge_top]
        to_judge += [p for p in cands if p not in to_judge and ("proposed by LLM" in p.notes or p.kind == "montage"
                                                                 or _source(p) != "ours")]
        res.log += J.judge_many(llm, to_judge, eps, batch=cfg.judge_batch,
                                progress=lambda d, n: say(f"LLM judged {d}/{n}", 0.65 + 0.3 * d / n))
        for p in to_judge:
            if "error" in p.llm:
                res.log.append(f"judge error on {rank.segment_str(p)}: {p.llm['error']}")
            p.scores["final"] = rank.final_score(p)
        res.log += llm.events
        # candidates the LLM never saw get a small penalty so judged ones are compared fairly
        for p in cands:
            if "hook_score" not in p.llm:
                p.scores["final"] = round(rank.final_score(p) * 0.9, 4)

    say("ranking", 0.9)
    srcs = {id(p): _source(p) for p in cands}
    ranked = rank.dedupe(cands)
    for p in ranked:   # who found this moment? (credit every pool whose clip overlapped it)
        found = {srcs[id(q)] for q in cands if q.kind == p.kind == "single"
                 and q.segments[0].episode == p.segments[0].episode
                 and C._iou((q.segments[0].start, q.segments[0].end), (p.segments[0].start, p.segments[0].end)) > 0.5}
        found.add(srcs[id(p)])
        p.notes.append("found by: " + ", ".join(sorted(found)))
    ranked = rank.diversify(ranked)
    res.qa_tournament = {}
    if llm.enabled and cfg.tournament:
        n_ev = len(llm.events)
        pool = ranked[:cfg.tournament]
        extra = [p for p in ranked[cfg.tournament:] if srcs[id(p)] != "ours"][:max(0, len(cfg.merge) * 8)]
        pool += extra
        in_pool = {id(p) for p in pool}
        rest = [p for p in ranked if id(p) not in in_pool]
        res.qa_tournament = T.run_tournament(
            llm, pool, rounds=cfg.rounds, recheck=cfg.recheck, label=rank.segment_str,
            progress=lambda d, n: say(f"head-to-head {d}/{n}", 0.9 + 0.06 * d / n))
        if "skipped" not in res.qa_tournament:
            T.blend(pool)
            before = [id(p) for p in pool]
            pool = rank.diversify(pool)
            res.qa_tournament["top10_changed"] = sum(1 for a, b in zip(before[:10], [id(p) for p in pool[:10]]) if a != b)
            ranked = pool + rest
        res.log += llm.events[n_ev:]
    for i, p in enumerate(ranked):
        p.id = f"P{i + 1:03d}"
    res.ranked = ranked
    res.plan = rank.posting_order(ranked[: max(cfg.top_n, 20)])

    say("quality checks", 0.98)
    bad = {p.id: qa.validate(p, eps) for p in ranked}
    bad = {k: v for k, v in bad.items() if v}
    res.qa = {"validator_failures": bad, "n_checked": len(ranked),
              "recall": qa.recall_report(ranked, cfg.reference, eps),
              "tournament": res.qa_tournament,
              "sources": _source_share(ranked)}
    res.meta = {
        "engine": llm.describe(), "provider": llm.provider, "model": llm.model,
        "episodes": len(show), "show_minutes": round(sum(e.duration for e in show) / 60, 1),
        "beats": sum(len(e.beats) for e in show), "candidates": len(cands), "after_dedupe": len(ranked),
        "llm_calls": llm.calls, "llm_cache_hits": llm.cache_hits,
        "tokens_in": llm.tokens_in, "tokens_out": llm.tokens_out,
        "seconds": round(time.time() - t_start, 1), "config": cfg.__dict__,
    }
    from .report import write_all
    res.out_path = write_all(res, cfg)
    say(f"done -> {res.out_path}", 1.0)
    return res
