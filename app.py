"""Streamlit UI:  streamlit run app.py

Upload a show's episode scripts (.txt files or one .zip), pick a model, get ranked promos.
Or open a previous run from outputs/ to browse it without re-running.
"""
from __future__ import annotations

import html
import io
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# work from the app's own folder so .env, data/, outputs/ and eval/ resolve the same way
# however the app was launched
APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)
sys.path.insert(0, str(APP_DIR))

from promo_finder.llm import DEFAULT_MODELS  # noqa: E402
from promo_finder.pipeline import RunConfig, run  # noqa: E402

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")

KEY_ENV = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY",
           "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
PINK, VIOLET, AMBER, TEAL = "#FF4D7E", "#9B6BFF", "#FFB547", "#3DD6C4"
VERDICT = {"post": ("POST IT", TEAL), "maybe": ("MAYBE", AMBER), "reject": ("JUDGE SAYS NO", "#8A8296")}


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def H(s: str) -> None:
    """Render raw HTML. Lines are stripped so markdown never turns indented HTML into a code block."""
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


# ------------------------------------------------------------------ style -----------------
H("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {font-family: 'Inter', sans-serif;}
.block-container {padding-top: 1.6rem; max-width: 1300px;}
.hero {position: relative; overflow: hidden; border-radius: 22px; padding: 30px 34px 26px;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(255,77,126,.35), transparent 60%),
              radial-gradient(900px 300px at 100% 100%, rgba(155,107,255,.35), transparent 60%), #15101F;
  border: 1px solid rgba(255,255,255,.08); margin-bottom: 18px;}
.hero:before, .hero:after {content: ""; position: absolute; left: 0; right: 0; height: 12px;
  background: repeating-linear-gradient(90deg, rgba(255,255,255,.13) 0 14px, transparent 14px 28px);}
.hero:before {top: 6px;} .hero:after {bottom: 6px;}
.hero h1 {font-family: 'Bebas Neue', sans-serif; font-size: 64px; letter-spacing: 2px; line-height: .95; margin: 8px 0 6px;
  background: linear-gradient(90deg, #fff, #FFC2D3 45%, #C9B2FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.hero p {color: #CFC6DD; font-size: 16px; max-width: 760px; margin: 0;}
.steps {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px;}
.step {background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1); border-radius: 999px;
  padding: 6px 14px; font-size: 13px; color: #EDE6F7;}
.step b {color: #FF8FAE; margin-right: 6px;}
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 6px 0 18px;}
.kpi {background: #1A1424; border: 1px solid rgba(255,255,255,.07); border-radius: 16px; padding: 14px 16px;}
.kpi .v {font-family: 'Bebas Neue', sans-serif; font-size: 38px; line-height: 1; color: #fff;}
.kpi .l {font-size: 12px; text-transform: uppercase; letter-spacing: 1.2px; color: #A79DB8; margin-top: 4px;}
.kpi .s {font-size: 12px; color: #FF8FAE; margin-top: 2px;}
.card {background: linear-gradient(180deg, #1C1528, #16111F); border: 1px solid rgba(255,255,255,.08);
  border-radius: 18px; padding: 18px 20px 14px; margin-bottom: 6px; position: relative;}
.card.spot {border: 1px solid rgba(255,77,126,.55); box-shadow: 0 0 40px rgba(255,77,126,.15);}
.top {display: flex; gap: 16px; align-items: flex-start;}
.rank {font-family: 'Bebas Neue', sans-serif; font-size: 46px; line-height: .9; min-width: 64px; text-align: center;
  background: linear-gradient(180deg, #FF4D7E, #9B6BFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.rank small {display: block; font-family: 'Inter'; font-size: 10px; letter-spacing: 1.5px; -webkit-text-fill-color: #A79DB8;}
.title {font-size: 18px; font-weight: 600; color: #fff; line-height: 1.35;}
.chips {display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 2px;}
.chip {font-size: 12px; padding: 3px 10px; border-radius: 999px; background: rgba(255,255,255,.07); color: #DCD3EA;
  font-family: 'JetBrains Mono', monospace;}
.pill {font-size: 11px; font-weight: 700; letter-spacing: 1px; padding: 3px 10px; border-radius: 999px; color: #0D0A14;}
.score {margin-left: auto; text-align: right; min-width: 92px;}
.score .n {font-family: 'Bebas Neue', sans-serif; font-size: 40px; line-height: .9; color: #fff;}
.score .t {font-size: 10px; letter-spacing: 1.5px; color: #A79DB8;}
.bars {display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 14px 0 10px;}
.bar .lab {display: flex; justify-content: space-between; font-size: 12px; color: #BDB3CC; margin-bottom: 4px;}
.bar .track {height: 7px; border-radius: 6px; background: rgba(255,255,255,.08); overflow: hidden;}
.bar .fill {height: 100%; border-radius: 6px;}
.ribbon {position: relative; height: 30px; margin: 10px 0 4px; border-radius: 8px; overflow: hidden;
  background: repeating-linear-gradient(90deg, rgba(255,255,255,.05) 0 2px, transparent 2px 10px), #120E1A;}
.seg {position: absolute; top: 0; bottom: 0; border-right: 2px solid #0D0A14; opacity: .85;}
.seg span {position: absolute; left: 6px; top: 6px; font-size: 11px; color: #0D0A14; font-weight: 700;
  font-family: 'JetBrains Mono', monospace; white-space: nowrap;}
.mark {position: absolute; top: -2px; bottom: -2px; width: 3px; background: #fff; box-shadow: 0 0 8px #fff;}
.ticks {display: flex; justify-content: space-between; font-size: 11px; color: #8F86A0; font-family: 'JetBrains Mono', monospace;}
.beats {display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 10px;}
.beat {background: rgba(255,255,255,.035); border-radius: 12px; padding: 10px 12px; font-size: 14px; color: #E9E2F3;}
.beat .k {font-size: 11px; letter-spacing: 1.4px; font-weight: 700; margin-bottom: 4px;}
.why {margin-top: 10px; font-size: 14px; color: #CFC6DD;}
.why b {color: #fff;}
.flag {display: inline-block; font-size: 12px; margin: 8px 6px 0 0; padding: 3px 10px; border-radius: 8px;
  background: rgba(255,181,71,.12); color: #FFCB7A; border: 1px solid rgba(255,181,71,.25);}
.sb {font-family: 'JetBrains Mono', monospace; font-size: 13px; line-height: 1.55;}
.sb .ln {display: grid; grid-template-columns: 70px 1fr; gap: 10px; padding: 3px 0; border-bottom: 1px dashed rgba(255,255,255,.05);}
.sb .tc {color: #8F86A0;}
.sb .tag {font-size: 10px; font-weight: 700; padding: 1px 6px; border-radius: 4px; margin-right: 6px;}
.sb .who {color: #FF8FAE; font-weight: 600;}
.sb .emo {color: #A79DB8; font-style: italic;}
.sb .sep {color: #C9B2FF; padding: 8px 0 4px; font-weight: 600;}
.day {background: #1A1424; border: 1px solid rgba(255,255,255,.07); border-radius: 16px; padding: 14px; height: 100%;}
.day .d {font-family: 'Bebas Neue', sans-serif; font-size: 28px; color: #FF8FAE; line-height: 1;}
.day .h {font-size: 14px; color: #fff; margin: 6px 0; font-weight: 600;}
.day .r {font-size: 12px; color: #A79DB8;}
.empty {text-align: center; padding: 40px; border: 1px dashed rgba(255,255,255,.15); border-radius: 18px; color: #BDB3CC;}
.how {display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 8px;}
.how div {background: #1A1424; border-radius: 16px; padding: 16px; border: 1px solid rgba(255,255,255,.07); font-size: 14px; color: #CFC6DD;}
.how b {display: block; font-family: 'Bebas Neue', sans-serif; font-size: 26px; color: #fff; letter-spacing: 1px; font-weight: 400;}
@media (max-width: 700px) {.hero h1 {font-size: 44px;} .bars, .beats {grid-template-columns: 1fr;} .score .n {font-size: 30px;}}
</style>
""")

H("""
<div class="hero">
<h1>🎬 Promo Finder</h1>
<p>Drop in a show's episode scripts. Get a ranked shortlist of 20&ndash;60 second promos that open on a hook
and cut on a cliffhanger, with exact cut points an editor can trust. Every timestamp comes from the script, never from the model.</p>
<div class="steps">
<span class="step"><b>01</b>Parse every line</span><span class="step"><b>02</b>Find clean cut points</span>
<span class="step"><b>03</b>Score hook &amp; cliffhanger</span><span class="step"><b>04</b>LLM judges blind</span>
<span class="step"><b>05</b>Rank &amp; plan the posts</span>
</div></div>
""")

# ------------------------------------------------------------------ sidebar ---------------
saved_runs = sorted(Path("outputs").glob("*/promos.json"), key=lambda p: p.stat().st_mtime, reverse=True)
with st.sidebar:
    st.markdown("### 🎛️ Control room")
    modes = ["Run on scripts", "Browse a saved run"]
    mode = st.radio("Mode", modes, index=1 if saved_runs else 0)
    st.divider()
    if mode == "Run on scripts":
        src = st.radio("Scripts", ["Upload", "Local folder"], horizontal=True)
        files = folder = None
        if src == "Upload":
            files = st.file_uploader("Episode .txt files or one .zip", type=["txt", "zip"], accept_multiple_files=True)
        else:
            folder = st.text_input("Folder", "data/input")
        st.markdown("#### 🧠 Model")
        prov_default = os.environ.get("PROMO_LLM_PROVIDER", "none")
        provs = ["none", "gemini", "groq", "openrouter", "ollama", "openai", "anthropic"]
        provider = st.selectbox("Provider", provs, index=provs.index(prov_default) if prov_default in provs else 0,
                                help="'none' = rules only: free, instant, deterministic, weaker ranking.")
        model = st.text_input("Model", os.environ.get("PROMO_LLM_MODEL") if provider == prov_default else DEFAULT_MODELS.get(provider, "")
                              ) if provider != "none" else ""
        if provider in KEY_ENV and not os.environ.get(KEY_ENV[provider]):
            k = st.text_input(f"{KEY_ENV[provider]}", type="password", help="Kept in memory for this session only.")
            if k:
                os.environ[KEY_ENV[provider]] = k
        if provider == "openai":
            base = st.text_input("Base URL (any OpenAI-compatible API)", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
            os.environ["OPENAI_BASE_URL"] = base
        judge_top = st.slider("Candidates the LLM judges", 10, 120, 60, 10, disabled=provider == "none")
        propose = st.checkbox("LLM also reads whole episodes", True, disabled=provider == "none")
        go = st.button("✨ Find promos", type="primary", width="stretch")
    st.caption("Scripts stay on this machine. Only the selected LLM provider sees clip text.")


# ------------------------------------------------------------------ helpers ---------------
def judge_scores(p: dict) -> tuple[list[tuple[str, float, str]], str]:
    j = p.get("llm") or {}
    if "hook_score" in j:
        return [("Hook", j["hook_score"] / 10, PINK), ("Cliffhanger", j["cliff_score"] / 10, VIOLET),
                ("Stands alone", j["standalone_score"] / 10, TEAL)], "LLM judge"
    s = p["scores"]
    return [("Hook", s["hook"], PINK), ("Cliffhanger", s["cliff"], VIOLET), ("Stands alone", s["standalone"], TEAL)], "rules"


def ribbon(p: dict) -> str:
    total = max(p["total_duration_s"], 0.1)
    cols = [PINK, VIOLET, AMBER, TEAL, "#6FA8FF"]
    out, x = [], 0.0
    for i, s in enumerate(p["segments"]):
        w = 100 * s["duration_s"] / total
        label = f"E{s['episode']} {s['start']}" if w > 14 else f"E{s['episode']}"
        out.append(f'<div class="seg" style="left:{x:.2f}%;width:{w:.2f}%;background:{cols[i % len(cols)]}"><span>{esc(label)}</span></div>')
        x += w
    if p.get("hook_at_s") is not None:
        out.append(f'<div class="mark" title="hook lands" style="left:{min(99, 100 * p["hook_at_s"] / total):.2f}%"></div>')
    return (f'<div class="ribbon">{"".join(out)}</div><div class="ticks"><span>0s · hook @{p.get("hook_at_s") or 0:.1f}s</span>'
            f'<span>cut ✂ {total:.0f}s</span></div>')


def card(p: dict, spotlight: bool = False) -> None:
    j = p.get("llm") or {}
    v = VERDICT.get(j.get("verdict", ""))
    pill = f'<span class="pill" style="background:{v[1]}">{v[0]}</span>' if v else ""
    kind = "🎞️ MONTAGE" if p["type"] == "montage" else "🎬 SINGLE SCENE"
    segs = "".join(f'<span class="chip">Ep {s["episode"]} · {s["start"]}–{s["end"]}</span>' for s in p["segments"])
    sc = p["scores"]
    if "h2h_wins" in sc:
        segs += (f'<span class="chip" style="background:rgba(255,77,126,.18);color:#FFC2D3">🥊 head-to-head '
                 f'{sc["h2h_wins"]}–{sc["h2h_losses"]}</span>')
    found = next((n[10:] for n in p.get("notes", []) if n.startswith("found by: ")), "")
    if found and found != "ours":
        segs += f'<span class="chip" style="background:rgba(61,214,196,.15);color:#9FF0E6">🔎 found by {esc(found)}</span>'
    bars, src = judge_scores(p)
    bar_html = "".join(f'<div class="bar"><div class="lab"><span>{n}</span><span>{v_ * 10:.1f}</span></div>'
                       f'<div class="track"><div class="fill" style="width:{100 * v_:.0f}%;background:{c}"></div></div></div>'
                       for n, v_, c in bars)
    flags = "".join(f'<span class="flag">⚠ {esc(f)}</span>' for f in p["flags"])
    who = "🤖 judge" if p.get("text_source") == "llm" else "📐 rules"
    H(f"""
<div class="card{' spot' if spotlight else ''}">
<div class="top">
<div class="rank"><small>RANK</small>#{p['rank']}</div>
<div style="flex:1">
<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:11px;letter-spacing:1.4px;color:#A79DB8">{kind} · {p['total_duration_s']:.0f}s {pill}</div>
<div class="title">{esc(p['hook'])}</div>
<div class="chips">{segs}</div>
</div>
<div class="score"><div class="n">{p['scores']['final']:.2f}</div><div class="t">FINAL SCORE</div></div>
</div>
{ribbon(p)}
<div class="bars">{bar_html}</div>
<div class="beats">
<div class="beat"><div class="k" style="color:{VIOLET}">✂ CUTS ON</div>{esc(p['cliffhanger'])}</div>
<div class="beat"><div class="k" style="color:{TEAL}">💡 WHY IT WORKS <span style="color:#8F86A0;font-weight:400;letter-spacing:0">({who})</span></div>{esc(p['why_it_works'])}</div>
</div>
<div class="why" style="font-size:12px;color:#8F86A0">Bars: {src} scores out of 10. White marker on the strip = where the hook lands.</div>
{flags}
</div>""")
    with st.expander("🎞️ Storyboard: every line in the clip"):
        H(storyboard(p["transcript"]))
        games = (p.get("llm") or {}).get("h2h") or []
        if games:
            st.markdown("**🥊 Head-to-head games** (blind A-vs-B, judged by the LLM)")
            H("".join(f'<div class="beat" style="margin:4px 0">{"✅ won" if g["won"] else "❌ lost"} '
                      f'vs <code>{esc(g["vs"])}</code> · round {g["round"]} · confidence {g["confidence"]}/3'
                      f'<br><span style="color:#A79DB8">{esc(g["reason"])}</span></div>' for g in games))
        st.json(p["scores"], expanded=False)


LINE_RE = re.compile(r"^(\S+) \[E\d+ \S+\] (.*)$")
TAG_COL = {"ACTION": "#6FA8FF", "SFX": AMBER, "CAMERA": "#8A8296"}


def storyboard(transcript: str) -> str:
    rows = []
    for line in transcript.splitlines():
        if line.startswith("---"):
            rows.append(f'<div class="sep">{esc(line.strip("- "))}</div>')
            continue
        m = LINE_RE.match(line)
        if not m:
            rows.append(f'<div class="ln"><span></span><span>{esc(line)}</span></div>')
            continue
        tc, body = m.groups()
        t = re.match(r"^\[([A-Z ]+)\]\s*(.*)$", body)
        d = re.match(r"^([^:(]{1,40}?)\s*(\(([^)]*)\))?:\s*(.*)$", body)
        if t:
            c = TAG_COL.get(t.group(1), "#8A8296")
            body_html = f'<span class="tag" style="background:{c};color:#0D0A14">{esc(t.group(1))}</span>{esc(t.group(2))}'
        elif d:
            body_html = (f'<span class="who">{esc(d.group(1))}</span> '
                         + (f'<span class="emo">({esc(d.group(3))})</span> ' if d.group(3) else "") + esc(d.group(4)))
        else:
            body_html = esc(body)
        rows.append(f'<div class="ln"><span class="tc">{esc(tc)}</span><span>{body_html}</span></div>')
    return f'<div class="sb">{"".join(rows)}</div>'


def posting_plan(promos: list[dict], n: int = 10) -> list[tuple[dict, str]]:
    """Same rule as promo_finder.rank.posting_order, on the saved JSON: early episodes first
    (a new viewer lands on episode 1), never two clips of the same episode back to back, ≤3 montages."""
    def bonus(p):
        last = max(s["episode"] for s in p["segments"])
        return 0.06 if last <= 5 else 0.03 if last <= 15 else -0.03 if last >= 40 else 0.0
    pool = sorted(promos, key=lambda p: -(p["scores"]["final"] + bonus(p)))
    plan, prev = [], set()
    while pool and len(plan) < n:
        pick = next((p for p in pool if not ({s["episode"] for s in p["segments"]} & prev)
                     and not (p["type"] == "montage" and sum(q["type"] == "montage" for q, _ in plan) >= 3)), pool[0])
        pool.remove(pick)
        last = max(s["episode"] for s in pick["segments"])
        why = ("sells the premise a new viewer starts on" if last <= 5 else
               "early story: little context, no big spoiler" if last <= 15 else
               "late-story payoff: test after premise clips" if last >= 40 else "mid-story escalation")
        plan.append((pick, why))
        prev = {s["episode"] for s in pick["segments"]}
    return plan


ALT_CFG = dict(background="transparent")


def themed(ch: alt.Chart) -> alt.Chart:
    return (ch.properties(**ALT_CFG)
            .configure_axis(labelColor="#BDB3CC", titleColor="#BDB3CC", gridColor="#2A2236", domainColor="#3A3048")
            .configure_legend(labelColor="#BDB3CC", titleColor="#BDB3CC")
            .configure_view(strokeWidth=0))


def show_results(data: dict, qa_md: str | None, folder: Path | None):
    meta, promos = data["meta"], data["promos"]
    posts = sum((p.get("llm") or {}).get("verdict") == "post" for p in promos)
    eng = meta["engine"]
    H(f"""
<div class="kpis">
<div class="kpi"><div class="v">{meta['episodes']}</div><div class="l">Episodes read</div><div class="s">{meta['show_minutes']} min of story</div></div>
<div class="kpi"><div class="v">{meta.get('beats', '–')}</div><div class="l">Script lines</div><div class="s">each with its timestamp</div></div>
<div class="kpi"><div class="v">{meta['candidates']}</div><div class="l">Clips considered</div><div class="s">{meta.get('after_dedupe', len(promos))} after de-dupe</div></div>
<div class="kpi"><div class="v">{len(promos)}</div><div class="l">Promos ranked</div><div class="s">{posts} marked "post" by judge</div></div>
<div class="kpi"><div class="v">{meta['seconds']:.0f}s</div><div class="l">Run time</div><div class="s">{esc(eng.split(':')[-1])}</div></div>
</div>""")
    st.caption(f"Engine {eng} · {meta['llm_calls']} LLM calls ({meta['llm_cache_hits']} cached) · "
               f"tokens in/out {meta['tokens_in']:,}/{meta['tokens_out']:,}")

    t1, t2, t3, t4, t5 = st.tabs(["🏆 Top picks", "🗺️ Story map", "📅 Posting plan", "📊 All promos", "✅ Quality check"])
    with t1:
        f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.6, 1])
        kind = f1.multiselect("Type", ["single", "montage"], ["single", "montage"])
        verdicts = f2.multiselect("Judge verdict", ["post", "maybe", "reject", "not judged"],
                                  ["post", "maybe", "not judged"])
        eps = f3.slider("Episodes", 1, meta["episodes"], (1, meta["episodes"]))
        n = f4.number_input("Show", 1, len(promos), min(10, len(promos)))
        q = st.text_input("🔎 Search hooks, cliffhangers, lines", placeholder="e.g. slap, poison, wife, ring…")
        sel = [p for p in promos if p["type"] in kind
               and ((p.get("llm") or {}).get("verdict") or "not judged") in verdicts
               and all(eps[0] <= s["episode"] <= eps[1] for s in p["segments"])
               and (not q or q.lower() in (p["hook"] + p["cliffhanger"] + p["why_it_works"] + p["transcript"]).lower())]
        if not sel:
            H('<div class="empty">No promo matches those filters. Try widening the episode range or verdicts.</div>')
        for i, p in enumerate(sel[:int(n)]):
            card(p, spotlight=(i == 0 and p["rank"] == 1))

    with t2:
        st.markdown("**Where the promo gold sits in the story.** Each bar is a clip: its episode, and where in the "
                    "episode it starts and ends. Brighter = higher score. Hover for the hook.")
        top = st.slider("Clips on the map", 10, len(promos), min(60, len(promos)), 5)
        rows = [{"episode": s["episode"], "from_min": s["start_s"] / 60, "to_min": s["end_s"] / 60,
                 "rank": p["rank"], "score": p["scores"]["final"], "type": p["type"], "hook": p["hook"][:110],
                 "clip": f"#{p['rank']} E{s['episode']} {s['start']}–{s['end']}"}
                for p in promos[:top] for s in p["segments"]]
        df = pd.DataFrame(rows)
        bars = alt.Chart(df).mark_bar(size=11, cornerRadius=3).encode(
            x=alt.X("episode:Q", title="Episode", scale=alt.Scale(domain=[0.5, meta["episodes"] + 0.5]), axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("from_min:Q", title="Minutes into the episode", scale=alt.Scale(reverse=True)), y2="to_min:Q",
            color=alt.Color("score:Q", title="Score", scale=alt.Scale(range=["#3B2A55", VIOLET, PINK, "#FFD1DD"])),
            opacity=alt.condition(alt.datum.type == "montage", alt.value(0.55), alt.value(0.95)),
            tooltip=["clip", "type", alt.Tooltip("score:Q", format=".2f"), "hook"]).properties(height=380)
        st.altair_chart(themed(bars), width="stretch")
        per = df.groupby("episode").size().reindex(range(1, meta["episodes"] + 1), fill_value=0).reset_index(name="clips")
        heat = alt.Chart(per).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=PINK).encode(
            x=alt.X("episode:O", title="Episode"), y=alt.Y("clips:Q", title="Clips from this episode"),
            tooltip=["episode", "clips"]).properties(height=170)
        st.altair_chart(themed(heat), width="stretch")
        st.caption("Montages (faded) stitch moments from several episodes into one promo.")

    with t3:
        st.markdown("**The first 10 posts, in order.** A viewer who taps the ad lands on episode 1, so clips that "
                    "sell the premise go first; the same episode is never posted twice in a row.")
        plan = posting_plan(promos)
        for row in range(0, len(plan), 5):
            cs = st.columns(5)
            for c, (k, (p, why)) in zip(cs, enumerate(plan[row:row + 5], start=row + 1)):
                eps_txt = " + ".join(f"E{s['episode']}" for s in p["segments"])
                c.markdown(" ".join(x.strip() for x in f"""
<div class="day"><div class="d">POST {k}</div>
<div class="chips"><span class="chip">#{p['rank']}</span><span class="chip">{eps_txt}</span><span class="chip">{p['total_duration_s']:.0f}s</span></div>
<div class="h">{esc(p['hook'][:120])}</div><div class="r">{esc(why)}</div></div>""".splitlines()), unsafe_allow_html=True)
            st.write("")

    with t4:
        df = pd.DataFrame([{"rank": p["rank"], "type": p["type"], "seconds": p["total_duration_s"],
                            "segments": " + ".join(f"E{s['episode']} {s['start']}-{s['end']}" for s in p["segments"]),
                            "verdict": (p.get("llm") or {}).get("verdict", "–"), "score": p["scores"]["final"],
                            "hook": p["hook"], "cliffhanger": p["cliffhanger"], "why": p["why_it_works"],
                            "flags": "; ".join(p["flags"])} for p in promos])
        st.dataframe(df, width="stretch", hide_index=True, height=520, column_config={
            "score": st.column_config.ProgressColumn("score", min_value=0, max_value=1, format="%.2f"),
            "seconds": st.column_config.NumberColumn("seconds", format="%.0f s")})
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇ CSV", df.to_csv(index=False).encode("utf-8-sig"), "promos.csv", width="stretch")
        d2.download_button("⬇ JSON", json.dumps(data, indent=2, ensure_ascii=False).encode(), "promos.json", width="stretch")
        md = folder / "promos.md" if folder else None
        if md and md.exists():
            d3.download_button("⬇ Editor report (.md)", md.read_bytes(), "promos.md", width="stretch")

    with t5:
        if qa_md:
            st.markdown(qa_md)
        else:
            st.info("No quality report for this run.")


# ------------------------------------------------------------------ main ------------------
if mode == "Browse a saved run":
    if not saved_runs:
        H('<div class="empty">No saved runs in <code>outputs/</code> yet. Switch to <b>Run on scripts</b> in the sidebar.</div>')
    else:
        pick = st.selectbox("🎞️ Saved run", saved_runs, format_func=lambda p: p.parent.name)
        qa = pick.parent / "qa_report.md"
        show_results(json.loads(pick.read_text(encoding="utf-8")), qa.read_text(encoding="utf-8") if qa.exists() else None, pick.parent)
else:
    if go:
        if src == "Upload":
            if not files:
                st.error("Upload at least one episode script.")
                st.stop()
            tmp = Path(tempfile.mkdtemp(prefix="promo_"))
            for f in files:
                if f.name.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(f.read())) as z:
                        for m in z.namelist():
                            if m.lower().endswith(".txt") and not m.startswith("__MACOSX"):
                                (tmp / Path(m).name).write_bytes(z.read(m))
                else:
                    (tmp / f.name).write_bytes(f.read())
            in_dir = str(tmp)
        else:
            in_dir = folder
        with st.status("🎬 Rolling… reading the scripts", expanded=True) as status:
            bar = st.progress(0.0, "starting")
            try:
                res = run(RunConfig(input_dir=in_dir, provider=provider, model=model or None, judge_top=judge_top,
                                    llm_propose=propose), progress=lambda m, f: bar.progress(min(1.0, f), m))
            except SystemExit as e:
                status.update(label="Run failed", state="error")
                st.error(str(e))
                st.stop()
            status.update(label="✅ That's a wrap: promos ranked", state="complete", expanded=False)
        if res.log:
            with st.expander(f"🧾 LLM log ({len(res.log)} events: model suggestions code repaired or dropped)"):
                st.text("\n".join(res.log))
        out = res.out_path
        st.success(f"Raw outputs saved to {out}")
        show_results(json.loads((out / "promos.json").read_text(encoding="utf-8")),
                     (out / "qa_report.md").read_text(encoding="utf-8"), out)
    else:
        H("""
<div class="how">
<div><b>1 · Load a show</b>Upload the episode <code>.txt</code> scripts or one <code>.zip</code> in the sidebar, or point at a folder.</div>
<div><b>2 · Pick a brain</b><i>none</i> = free rules engine in seconds. Add Gemini, Groq, OpenRouter, Ollama or any OpenAI-style API for a blind LLM judge.</div>
<div><b>3 · Find promos</b>Hit <b style="display:inline;font-family:Inter;font-size:14px;color:#FF8FAE">✨ Find promos</b>. Every cut lands on a shot change, never mid-line.</div>
<div><b>4 · Ship it</b>Browse ranked cards, the story map and a 10-post plan; download CSV, JSON and the editor report.</div>
</div>""")
