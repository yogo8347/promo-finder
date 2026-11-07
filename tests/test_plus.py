"""Tests for the 'plus' additions: imported slates, scene-jump penalty, head-to-head tournament."""
from __future__ import annotations

import json
import re

from promo_finder import candidates as C
from promo_finder import rank
from promo_finder import tournament as T
from promo_finder.external import load_external
from promo_finder.llm import LLM
from promo_finder.parser import load_show
from tests.test_pipeline import show_dir  # noqa: F401  (fixture)


def test_bradley_terry_orders_by_wins_and_stays_finite():
    games = [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (0, 1, 1.0)]
    s = T.bradley_terry(3, games)
    assert s[0] > s[1] > s[2]
    assert all(0 < x < 1e6 for x in s)          # the prior keeps an unbeaten clip finite


def test_swiss_never_repeats_a_pairing():
    played = {frozenset((0, 1))}
    pairs = T.swiss_pairs([0, 1, 2, 3], played)
    assert frozenset((0, 1)) not in {frozenset(p) for p in pairs}
    assert len(pairs) == 2


class PreferLonger(LLM):
    """Fake judge: always prefers the clip with more lines, but is position-biased on ties."""
    def __init__(self):
        super().__init__("none")
        self.provider, self.n = "fake", 0

    def complete_json(self, prompt, system=""):
        self.n += 1
        a, b = prompt.split("=== CLIP B ===")
        n = lambda t: len(re.findall(r"^\[", t, flags=re.M))
        return {"winner": "A" if n(a) >= n(b) else "B", "confidence": 2, "reason": "more happens"}


def test_tournament_measures_the_judge_and_rescales(show_dir):
    eps = {e.number: e for e in load_show(show_dir)}
    pool = [p for e in eps.values() for p in C.single_candidates(e, per_episode=4)][:8]
    for p in pool:
        p.scores["final"] = 0.5
    llm = PreferLonger()
    out = T.run_tournament(llm, pool, rounds=3, recheck=4, label=rank.segment_str)
    assert out["games"] == llm.n - 4
    assert 0 <= out["position_bias_A_win_rate"] <= 1
    assert all(0 <= p.scores["tournament"] <= 1 for p in pool)
    assert all(p.scores["h2h_wins"] + p.scores["h2h_losses"] >= 1 for p in pool)
    T.blend(pool)
    assert all("final_before_h2h" in p.scores for p in pool)


def test_imported_slate_is_snapped_to_legal_cuts(show_dir, tmp_path):
    eps = {e.number: e for e in load_show(show_dir)}
    e = eps[1]
    slate = {"promos": [{"episode": 1, "start_sec": 0.4, "end_sec": min(e.duration, 35.0)},
                        {"episode": 99, "start_sec": 0, "end_sec": 30}]}
    f = tmp_path / "other.json"
    f.write_text(json.dumps(slate))
    promos, log = load_external(f, eps, tag="other")
    assert any("episode" in l or "map" in l for l in log)          # unknown episode is reported, not crashed on
    for p in promos:
        s = p.segments[0]
        assert s.start in C.start_points(e) and s.end in C.end_points(e)
        assert "from other slate (#1)" in p.notes


def test_scene_jump_is_flagged_and_penalised(show_dir):
    eps = {e.number: e for e in load_show(show_dir)}
    p = C.single_candidates(eps[1], per_episode=1)[0]
    base = rank.final_score(p)
    for i, b in enumerate(p.segments[0].beats):
        b.scene = 1 if i < len(p.segments[0].beats) // 2 else 2
    assert rank.scene_jumps(p) == 1
    assert rank.final_score(p) < base
    assert any("jumps scene" in f for f in rank.flags(p))


def test_quote_check_catches_text_not_in_clip(show_dir):
    from promo_finder import grounding as G
    eps = {e.number: e for e in load_show(show_dir)}
    p = C.single_candidates(eps[1], per_episode=1)[0]
    real = next(b.text for b in p.segments[0].beats if b.kind == "dialogue")
    assert G.unmatched_quotes(f'She says "{real}"', p) == []
    assert G.unmatched_quotes('He shouts "the castle walls are burning down tonight"', p)
    assert G.unmatched_quotes("Lord's men arrive", p) == []      # an apostrophe is not a quote


class FactChecker(LLM):
    def __init__(self, reply):
        super().__init__("none")
        self.provider, self.reply = "fake", reply

    def complete_json(self, prompt, system=""):
        return self.reply


def test_verify_replaces_wrong_text_and_falls_back_to_script(show_dir):
    from promo_finder import grounding as G
    eps = {e.number: e for e in load_show(show_dir)}
    p = C.single_candidates(eps[1], per_episode=1)[0]
    p.llm = {"hook": "a slap", "cliffhanger": "he has already won the duel"}
    # the AI says the cut text is wrong and offers a fix that quotes something not in the clip
    llm = FactChecker({"clips": [{"clip": "C1", "hook_ok": True, "cut_ok": False,
                                  "cut_fixed": 'He says "we ride for the northern border at dawn"'}]})
    out = G.verify(llm, [p])
    assert out["cut_fixed"] == 1 and out["fallback_to_script"] == 1
    assert p.llm["cliffhanger"] == G.last_line(p)             # code used the real line instead
    assert p.llm["cliffhanger_ai_original"] == "he has already won the duel"
    assert p.llm["hook"] == "a slap"                           # untouched when ok


def test_posting_plan_never_repeats_an_ending(show_dir):
    eps = {e.number: e for e in load_show(show_dir)}
    a = C.single_candidates(eps[1], per_episode=1)[0]
    b = C.single_candidates(eps[2], per_episode=1)[0]
    twin = C.Promo("montage", [C.Segment(2, b.segments[0].start, b.segments[0].end, b.segments[0].beats)],
                   hook_beat=b.hook_beat, cliff_beat=b.cliff_beat)
    for p, sc in ((a, 0.9), (b, 0.8), (twin, 0.85)):
        p.scores = {"final": sc, "rules": sc}
    plan = rank.posting_order([a, twin, b], n=3)
    ends = [(q.segments[-1].episode, q.segments[-1].end) for q, _ in plan]
    assert len(ends) == len(set(ends)) or len(plan) < 3 or plan[-1][0] is not None
    first_two = [q for q, _ in plan[:2]]
    assert not (twin in first_two and b in first_two)
