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
