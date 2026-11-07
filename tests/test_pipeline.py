"""Run:  python -m pytest -q

Synthetic scripts only (the real input pack is confidential and not in the repo)."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from promo_finder import candidates as C
from promo_finder import judge as J
from promo_finder import qa
from promo_finder.llm import LLM, LLMError, parse_json
from promo_finder.parser import fmt, load_show, parse_episode, to_seconds


def block(head, t0, t1, text):
    return f"**{head}:**\nStart: ({fmt(t0)}) | End: ({fmt(t1)})\n{text}\n"


def make_episode(n: int, extra_lines: int = 14) -> str:
    """A ~60 s fake episode: a slap hook, an argument, a cliffhanger sting at the end."""
    out = [f"EPISODE NUMBER: Episode {n}", f"EPISODE LOGLINE: Test episode {n}.", "CONTINUITY NOTES: none",
           "", "--- SCRIPT START ---", "", "SCENE 01", "Scene heading: INT. HALL - DAY",
           "Scene start: 00:00.00", "Scene end: 01:00.00", "", "Actions and dialogue:", ""]
    t = 0.0
    out.append(block("CAMERA", t, t + 3, "Close-up on Mia."))
    out.append(block("ACTION", t + 0.5, t + 2.5, "Mia slaps Jack across the face."))
    out.append(block("MIA (Furious, Shouting)", t + 1.0, t + 2.8, '"You married my sister?!"'))
    t = 3.0
    for i in range(extra_lines):
        out.append(block("CAMERA", t, t + 4, f"Shot {i}."))
        who = "JACK (Cold)" if i % 2 else "MIA (Desperate)"
        out.append(block(who, t + 0.3, t + 3.6, f'"Line number {i}, and you know it."'))
        t += 4
    out.append(block("CAMERA", t, t + 3, "Doors burst open."))
    out.append(block("JACK (Shocked)", t + 0.2, t + 2.0, '"Who are you?"'))
    out.append(block("SOUND FX", t + 1.5, t + 3, "Ominous drone swells to a cliffhanger sting."))
    out.append("--- SCRIPT END ---")
    return "\n".join(out)


@pytest.fixture
def show_dir(tmp_path: Path) -> Path:
    for n in (1, 2, 3):
        (tmp_path / f"episode-{n:03d}.txt").write_text(make_episode(n), encoding="utf-8")
    return tmp_path


def test_timestamps():
    assert to_seconds("01:39.10") == pytest.approx(99.1)
    assert to_seconds("1:02:03.5") == pytest.approx(3723.5)
    assert fmt(99.1) == "01:39.10"


def test_parse_counts_every_block(show_dir):
    ep = parse_episode(show_dir / "episode-001.txt")
    raw = (show_dir / "episode-001.txt").read_text()
    assert len(ep.beats) == raw.count("Start: (")
    d = [b for b in ep.beats if b.kind == "dialogue"][0]
    assert d.speaker == "MIA" and "Furious" in d.tags


def test_missing_episode_header_uses_filename(tmp_path):
    txt = make_episode(9).replace("EPISODE NUMBER: Episode 9\n", "")
    (tmp_path / "ep_07_final.txt").write_text(txt, encoding="utf-8")
    assert load_show(tmp_path)[0].number == 7


def test_honorifics_merge_only_when_safe(tmp_path):
    a = make_episode(1).replace("MIA (Desperate)", "LADY MIA (Desperate)")
    b = make_episode(2).replace("JACK (Cold)", "LORD ASH (Cold)").replace("MIA (Desperate)", "LADY ASH (Desperate)")
    (tmp_path / "e1.txt").write_text(a, encoding="utf-8")
    (tmp_path / "e2.txt").write_text(b, encoding="utf-8")
    speakers = {x.speaker for e in load_show(tmp_path) for x in e.beats if x.kind == "dialogue"}
    assert "LADY MIA" not in speakers            # merged into MIA, who also speaks bare
    assert {"LORD ASH", "LADY ASH"} <= speakers   # two different people stay apart


def test_candidates_obey_rules(show_dir):
    show = load_show(show_dir)
    eps = {e.number: e for e in show}
    promos = [p for e in show for p in C.single_candidates(e)] + C.montage_candidates(show)
    assert promos, "expected candidates"
    for p in promos:
        assert qa.validate(p, eps) == [], (p.segments, qa.validate(p, eps))


def test_cut_never_inside_a_line(show_dir):
    ep = load_show(show_dir)[0]
    for t in C.start_points(ep) + C.end_points(ep):
        for b in ep.beats:
            if b.kind == "dialogue":
                assert not (b.start < t - 0.25 and b.end > t + 0.25)


def test_hook_found_in_first_seconds(show_dir):
    ep = load_show(show_dir)[0]
    best = C.single_candidates(ep)[0]
    assert best.hook_beat.start - best.segments[0].start <= 3.0


class FakeLLM(LLM):
    """Returns canned JSON so we can test how model output is checked."""
    def __init__(self, reply):
        super().__init__("none")
        self.provider = "fake"
        self.reply = reply

    def complete(self, prompt, system="", want_json=True):
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


def test_hallucinated_line_ids_are_dropped(show_dir):
    ep = load_show(show_dir)[0]
    llm = FakeLLM({"promos": [{"start": "L999", "end": "L1000", "hook": "x", "cliffhanger": "y", "why": "z"}]})
    props, log = J.propose(llm, ep)
    assert props == [] and "hallucinated" in log[0]


def test_too_short_proposal_is_repaired_and_logged(show_dir):
    ep = load_show(show_dir)[0]
    eps = {ep.number: ep}
    llm = FakeLLM({"promos": [{"start": "L1", "end": "L2", "hook": "h", "cliffhanger": "c"}]})
    props, log = J.propose(llm, ep)
    assert len(props) == 1 and "repaired" in log[0]
    assert 20 <= props[0].duration <= 60 and qa.validate(props[0], eps) == []
    assert props[0].llm["proposal"]["hook"] == ""      # model text dropped once the cut changed


def test_valid_proposal_maps_to_script_times(show_dir):
    ep = load_show(show_dir)[0]
    llm = FakeLLM({"promos": [{"start": "L1", "end": "L12", "hook": "slap", "cliffhanger": "?", "why": "w"}]})
    props, _ = J.propose(llm, ep)
    assert len(props) == 1
    s = props[0].segments[0]
    assert s.start == 0.0 and 20 <= s.dur <= 60


def test_judge_scores_are_clamped_and_bad_json_is_survived(show_dir):
    show = load_show(show_dir)
    eps = {e.number: e for e in show}
    p = C.single_candidates(show[0])[0]
    J.judge(FakeLLM({"hook_score": 42, "cliffhanger_score": -3, "standalone_score": "7", "verdict": "POST"}), p, eps)
    assert p.llm["hook_score"] == 10 and p.llm["cliff_score"] == 0 and p.llm["verdict"] == "post"
    p2 = C.single_candidates(show[1])[0]
    J.judge(FakeLLM("sorry, I cannot"), p2, eps)
    assert "error" in p2.llm


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 2} hope that helps') == {"a": 2}
    with pytest.raises(LLMError):
        parse_json("no json here")


def test_full_run_rules_only(show_dir, tmp_path):
    from promo_finder.pipeline import RunConfig, run
    res = run(RunConfig(input_dir=str(show_dir), out_dir=str(tmp_path / "out"), provider="none",
                        reference=str(tmp_path / "none.json")), progress=lambda m, f: None)
    assert res.ranked and not res.qa["validator_failures"]
    for f in ("promos.json", "promos.csv", "promos.md", "qa_report.md", "run_meta.json"):
        assert (res.out_path / f).exists()


def test_reference_for_other_show_is_skipped(show_dir, tmp_path):
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"show": "Other", "fingerprint": "nothing like this", "picks": []}))
    show = load_show(show_dir)
    assert "skipped" in qa.recall_report([], ref, {e.number: e for e in show})


def test_batch_judge_matches_clips_and_falls_back(show_dir):
    show = load_show(show_dir)
    eps = {e.number: e for e in show}
    ps = [C.single_candidates(e)[0] for e in show]
    good = {"clips": [{"clip": f"C{i}", "hook_score": i, "cliffhanger_score": 5, "standalone_score": 5,
                       "verdict": "maybe"} for i in (1, 2, 3)]}
    J.judge_many(FakeLLM(good), ps, eps, batch=5)
    assert [p.llm["hook_score"] for p in ps] == [1, 2, 3]
    ps2 = [C.single_candidates(e)[1] for e in show]
    log = J.judge_many(FakeLLM({"clips": [{"clip": "C1"}]}), ps2, eps, batch=5)   # wrong count
    assert "one by one" in log[0] and all("hook_score" in p.llm for p in ps2)
