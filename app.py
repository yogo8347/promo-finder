"""Streamlit UI:  streamlit run app.py

A thin front-end over the same pipeline as the CLI (nothing here re-implements it):
  1. pick the screenplays (upload .txt / .zip, or a folder)
  2. pick the model and key in the sidebar (save it, test it, list the models your key can use)
  3. run, then read the slate, the posting plan and the quality checks
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# work from the app's own folder so .env, data/, outputs/ and eval/ resolve however it was launched
APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)
sys.path.insert(0, str(APP_DIR))

from promo_finder.llm import DEFAULT_MODELS, LLM  # noqa: E402  (also loads .env)
from promo_finder.pipeline import RunConfig, run  # noqa: E402

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")

ENV_KEYS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "ollama": None, "none": None}
PROVIDER_HELP = {
    "none": "Rules only. Free, about 30 s, no key, no internet. Weaker ranking.",
    "gemini": "Google AI Studio key (free). Default model: Gemma 4 26B.",
    "groq": "Groq key (free tier).", "openrouter": "OpenRouter key (has free models).",
    "openai": "OpenAI key, or ANY OpenAI-compatible API via the base URL below.",
    "anthropic": "Anthropic key.", "ollama": "A local Ollama model, no key.",
}
SPEED = {
    "Quick (about 5-10 min)": dict(judge_top=30, llm_propose=False, tournament=16, rounds=3, recheck=6),
    "Full (about 35 min, best)": dict(judge_top=60, llm_propose=True, tournament=28, rounds=5, recheck=10),
}

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
h1, h2, h3 {letter-spacing: .2px;}
h3 {color: #E3B341;}
div[data-testid="stMetric"] {background: #1F1F1F; border: 1px solid #2C2C2C; border-radius: 10px; padding: 10px 14px;}
div[data-testid="stMetricValue"] {color: #F0EBDD;}
.note {color: #A8A39A; font-size: 0.9rem;}
.clip {background: #1F1F1F; border-left: 4px solid #E3B341; border-radius: 8px; padding: 12px 16px; margin: 6px 0;}
.clip b {color: #E3B341;}
.banner {display: flex; gap: 18px; align-items: flex-start; padding: 20px 24px; margin-bottom: 8px; border-radius: 14px;
  border: 1px solid #3A3220; border-left: 5px solid #E3B341;
  background: linear-gradient(110deg, rgba(227,179,65,.16) 0%, rgba(227,179,65,.05) 45%, #1A1A1A 100%);}
.banner .mark {flex: 0 0 auto; width: 52px; height: 52px; border-radius: 12px; background: #E3B341; color: #141414;
  font-size: 24px; display: flex; align-items: center; justify-content: center; margin-top: 4px;}
.banner .name {font-size: 2.3rem; font-weight: 700; color: #F0EBDD; line-height: 1.1;}
.banner .name span {color: #E3B341;}
.banner .tag {color: #C9C3B6; font-size: .95rem; margin-top: 6px; max-width: 900px;}
.banner .chips {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;}
.banner .chips span {font-size: .82rem; padding: 4px 12px; border-radius: 999px; background: rgba(227,179,65,.12);
  border: 1px solid rgba(227,179,65,.35); color: #EBD9A8;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers -------------------
def save_key(name: str, value: str) -> None:
    """Write or replace one key in the local .env (git-ignored, never committed)."""
    path = APP_DIR / ".env"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    lines = [l for l in lines if not l.strip().startswith(f"{name}=")] + [f"{name}={value}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[name] = value


def gemini_models(key: str) -> list[str]:
    """models.list does not use your daily request quota."""
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                     headers={"x-goog-api-key": key}, timeout=30)
    r.raise_for_status()
    out = []
    for m in r.json().get("models", []):
        name = m["name"].replace("models/", "")
        if "generateContent" in m.get("supportedGenerationMethods", []) and not any(
                x in name for x in ("embed", "tts", "image", "aqa", "vision", "audio", "live")):
            out.append(name)
    return sorted(out)


def test_model(provider: str, model: str) -> tuple[bool, str]:
    """One tiny real call (no cache). Costs one request."""
    t0 = time.time()
    try:
        llm = LLM(provider, model or None, cache_dir=tempfile.mkdtemp(prefix="probe_"))
        llm.fallbacks = []
        llm.complete(f'Reply with JSON {{"ok": true, "t": {int(t0)}}}', want_json=True)
        return True, f"{llm.model} answered in {time.time() - t0:.1f}s"
    except Exception as e:  # show the provider's message as-is
        return False, f"{type(e).__name__}: {str(e)[:220]}"


def unpack_uploads(files) -> tuple[str | None, int]:
    if not files:
        return None, 0
    tmp = Path(tempfile.mkdtemp(prefix="promo_in_"))
    n = 0
    for f in files:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z:
                for m in z.namelist():
                    if m.lower().endswith(".txt") and "__MACOSX" not in m:
                        (tmp / Path(m).name).write_bytes(z.read(m))
                        n += 1
        else:
            (tmp / f.name).write_bytes(f.getvalue())
            n += 1
    return str(tmp), n


def posting_plan(promos: list[dict], n: int = 10) -> list[tuple[dict, str]]:
    """Same rule as promo_finder.rank.posting_order, applied to the saved JSON."""
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
        why = ("Episodes 1-5: sells the story a new viewer is about to start." if last <= 5 else
               "Early in the story: little context needed, nothing big spoiled." if last <= 15 else
               "Late-story payoff: strong, but post it after the premise clips." if last >= 40 else
               "Mid-story escalation.")
        plan.append((pick, why))
        prev = {s["episode"] for s in pick["segments"]}
    return plan


def segs(p: dict) -> str:
    return " + ".join(f"Ep {s['episode']} {s['start']}–{s['end']}" for s in p["segments"])


# ------------------------------------------------------------------ sidebar: model -----------
with st.sidebar:
    st.header("Model")
    provs = list(ENV_KEYS)
    default_prov = os.environ.get("PROMO_LLM_PROVIDER", "gemini")
    provider = st.selectbox("Provider", provs, provs.index(default_prov) if default_prov in provs else 0)
    st.caption(PROVIDER_HELP[provider])

    env_key = ENV_KEYS[provider]
    current = os.environ.get(env_key, "") if env_key else ""
    if env_key:
        with st.expander("API key", expanded=not current):
            if current:
                st.success(f"Key loaded from .env (ends …{current[-4:]})")
            else:
                st.warning("No key yet. Paste one below and press Save.")
            new_key = st.text_input(env_key, type="password", placeholder="Paste a key to add or change it",
                                    help="Saved only in the local .env file, which git ignores.")
            if st.button("Save key", width="stretch", disabled=not new_key):
                save_key(env_key, new_key.strip())
                st.rerun()

    if provider == "openai":
        os.environ["OPENAI_BASE_URL"] = st.text_input(
            "Base URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            help="Point this at any OpenAI-compatible API.")

    model = ""
    if provider != "none":
        default_model = os.environ.get("PROMO_LLM_MODEL") if provider == default_prov else None
        default_model = default_model or DEFAULT_MODELS.get(provider, "")
        if provider == "gemini" and current:
            if st.button("Show models my key can use", width="stretch",
                         help="Free: listing models does not use your daily quota."):
                try:
                    st.session_state["gemini_models"] = gemini_models(current)
                except Exception as e:
                    st.error(f"Could not list models: {e}")
        opts = st.session_state.get("gemini_models") if provider == "gemini" else None
        if opts:
            model = st.selectbox("Model", opts, opts.index(default_model) if default_model in opts else 0)
        else:
            model = st.text_input("Model", default_model)
        if st.button("Test this model", width="stretch", disabled=bool(env_key) and not current):
            with st.spinner("Calling the model once…"):
                ok, msg = test_model(provider, model)
            (st.success if ok else st.error)(msg)

    st.divider()
    st.header("Speed")
    speed = st.radio("How thorough?", list(SPEED), disabled=provider == "none",
                     help="Quick is right for a live demo. Full is what produced the submitted results.")
    with st.expander("Advanced"):
        s = dict(SPEED[speed])
        s["judge_top"] = st.slider("Clips the model scores", 10, 120, s["judge_top"], 10)
        s["llm_propose"] = st.checkbox("Model also reads whole episodes", s["llm_propose"])
        s["tournament"] = st.slider("Head-to-head pool (0 = off)", 0, 40, s["tournament"], 4)
        s["rounds"] = st.slider("Head-to-head rounds", 1, 7, s["rounds"])

# ------------------------------------------------------------------ main: input --------------
st.markdown("""
<div class="banner">
<div class="mark">▶</div>
<div>
<div class="name">Promo <span>Finder</span></div>
<div class="tag">Finds the 20–60 s clips worth posting from a microdrama's scripts: exact cut points, the hook,
the cliffhanger, and why it works. Every timestamp comes from the script, never from the model.</div>
<div class="chips"><span>⚡ Hook in the first 3 s</span><span>✂ Cuts on a cliffhanger</span><span>🎯 Makes sense to a stranger</span></div>
</div>
</div>
""".replace("\n", " "), unsafe_allow_html=True)

st.subheader("1. Screenplays")
src = st.radio("Input", ["Upload files", "Folder on this computer"], horizontal=True, label_visibility="collapsed")
workdir, n_files = None, 0
if src == "Upload files":
    ups = st.file_uploader("Episode .txt files, or one .zip of them", type=["txt", "zip"], accept_multiple_files=True)
    workdir, n_files = unpack_uploads(ups)
else:
    folder = st.text_input("Folder of .txt screenplays", st.session_state.get("folder", "data/input"))
    if folder and Path(folder).is_dir():
        workdir = folder
        st.session_state["folder"] = folder
        n_files = len(list(Path(folder).rglob("*.txt")))
    elif folder:
        st.error("That folder doesn't exist.")
if n_files:
    st.success(f"{n_files} screenplay file(s) ready.")
ep_spec = st.text_input("Episodes (optional)", placeholder="e.g. 1-5, or 1,3,7. Leave blank for all.")

st.subheader("2. Find promos")
c1, c2 = st.columns([1, 3])
go = c1.button("▶ Run", type="primary", disabled=not workdir, width="stretch")
c2.markdown(f"<div class='note'>Model: <b>{provider}{(' · ' + model) if model else ''}</b>"
            f"{'' if provider == 'none' else ' · ' + speed}</div>", unsafe_allow_html=True)

saved = sorted(Path("outputs").glob("*/promos.json"), key=lambda p: p.stat().st_mtime, reverse=True)
with st.expander("…or open a saved run"):
    if saved:
        names = [p.parent.name for p in saved]
        def _open():
            if st.session_state["saved_pick"] != "(choose)":
                st.session_state["run_dir"] = str(Path("outputs") / st.session_state["saved_pick"])
        st.selectbox("Saved run", ["(choose)"] + names, key="saved_pick", on_change=_open,
                     help="02_plus_gemma4_26b is the submitted result.")
    else:
        st.caption("No saved runs yet.")

if go and workdir:
    with st.status("Running…", expanded=True) as status:
        bar = st.progress(0.0)
        last = {"step": ""}

        def progress(msg: str, frac: float) -> None:
            step = msg.split(":")[0].split(" ")[0:2]
            step = " ".join(step)
            if step != last["step"]:
                st.write(f"• {msg}")
                last["step"] = step
            bar.progress(min(1.0, frac), text=msg)

        cfg = RunConfig(input_dir=workdir, provider=provider, model=model or None, episodes=ep_spec,
                        run_name=f"app_{time.strftime('%Y%m%d_%H%M%S')}",
                        **({k: v for k, v in s.items()} if provider != "none" else {}))
        try:
            res = run(cfg, progress=progress)
        except SystemExit as e:
            status.update(label=str(e), state="error")
            st.stop()
        except Exception as e:
            status.update(label=f"Run failed: {e}", state="error")
            st.exception(e)
            st.stop()
        status.update(label=f"Done: {len(res.ranked)} promos · {res.meta['engine']}", state="complete", expanded=False)
    st.session_state["run_dir"] = str(res.out_path)

# ------------------------------------------------------------------ results -------------------
run_dir = st.session_state.get("run_dir")
if not run_dir:
    st.info("Pick the screenplays above and press **▶ Run**, or open a saved run.")
    st.stop()

out = Path(run_dir)
data = json.loads((out / "promos.json").read_text(encoding="utf-8"))
meta, promos = data["meta"], data["promos"]
rm = json.loads((out / "run_meta.json").read_text(encoding="utf-8")) if (out / "run_meta.json").exists() else {}
qa = rm.get("qa", {})

st.subheader(f"3. Results  ·  {out.name}")
st.caption(f"{meta['engine']} · {meta['seconds']:.0f}s · {meta['llm_calls']} model calls "
           f"({meta['llm_cache_hits']} cached) · tokens in/out {meta['tokens_in']:,}/{meta['tokens_out']:,}")

tab_slate, tab_plan, tab_q, tab_how = st.tabs(["The slate", "Posting plan", "Quality checks", "How it works"])

with tab_slate:
    a, b, c, d = st.columns(4)
    a.metric("Promos found", len(promos))
    b.metric("Episodes read", meta["episodes"])
    c.metric("Clips considered", meta["candidates"])
    d.metric("Broken cuts", len(qa.get("validator_failures", {})))

    hide_ctx = st.checkbox("Hide clips flagged 'needs context'", value=False)
    show = [p for p in promos if not (hide_ctx and any("needs context" in f for f in p["flags"]))]
    st.dataframe(pd.DataFrame([{
        "#": p["rank"], "Clip": segs(p), "Length": f"{p['total_duration_s']:.0f}s",
        "Score": p["scores"]["final"], "Hook": p["hook"], "Cliffhanger": p["cliffhanger"],
        "Why it works": p["why_it_works"]} for p in show]),
        width="stretch", hide_index=True, height=360,
        column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.2f")})

    st.markdown("#### Look at one promo")
    if show:
        i = st.selectbox("Promo", range(len(show)), label_visibility="collapsed",
                         format_func=lambda k: f"#{show[k]['rank']} · {segs(show[k])} · {show[k]['total_duration_s']:.0f}s "
                                               f"· score {show[k]['scores']['final']:.2f}")
        p = show[i]
        j = p.get("llm") or {}
        m1, m2, m3, m4 = st.columns(4)
        if "hook_score" in j:
            m1.metric("Hook", f"{j['hook_score']:.0f}/10")
            m2.metric("Cliffhanger", f"{j['cliff_score']:.0f}/10")
            m3.metric("Makes sense alone", f"{j['standalone_score']:.0f}/10")
        else:
            m1.metric("Hook", f"{p['scores']['hook'] * 10:.1f}/10")
            m2.metric("Cliffhanger", f"{p['scores']['cliff'] * 10:.1f}/10")
            m3.metric("Makes sense alone", f"{p['scores']['standalone'] * 10:.1f}/10")
        sc = p["scores"]
        m4.metric("Head-to-head", f"{sc['h2h_wins']}–{sc['h2h_losses']}" if "h2h_wins" in sc else "—")
        st.markdown(f"<div class='clip'><b>Hook</b> (lands at {p['hook_at_s'] or 0:.1f}s): {p['hook']}<br>"
                    f"<b>Cuts on</b>: {p['cliffhanger']}<br><b>Why it works</b>: {p['why_it_works']}</div>",
                    unsafe_allow_html=True)
        if p["flags"]:
            st.warning("Check: " + " · ".join(p["flags"]))
        games = j.get("h2h") or []
        if games:
            with st.expander(f"Head-to-head games ({len(games)})"):
                for g in games:
                    st.write(f"{'✅ won' if g['won'] else '❌ lost'} vs {g['vs']} (round {g['round']}): {g['reason']}")
        st.markdown("**The clip, line by line** (left: time in the promo · bracket: episode time)")
        st.code(p["transcript"], language=None)

with tab_plan:
    st.markdown("**Post these first, in this order.** Someone who taps the ad lands on episode 1, so clips that "
                "sell the premise go early, and the same episode is never posted twice in a row.")
    for k, (p, why) in enumerate(posting_plan(promos), 1):
        st.markdown(f"<div class='clip'><b>{k}.</b> {segs(p)} · {p['total_duration_s']:.0f}s · rank #{p['rank']}<br>"
                    f"{p['hook']}<br><span class='note'>{why}</span></div>", unsafe_allow_html=True)

with tab_q:
    fails = qa.get("validator_failures", {})
    q1, q2, q3 = st.columns(3)
    q1.metric("Promos checked", qa.get("n_checked", len(promos)))
    q2.metric("Broken cuts", len(fails))
    t = qa.get("tournament") or {}
    q3.metric("Judge consistency (A/B swapped)", t.get("swap_recheck", "—"))
    if fails:
        st.error("These break a rule:")
        for pid, errs in fails.items():
            st.write(f"- {pid}: " + "; ".join(errs))
    else:
        st.success("Every promo is 20–60 s, its hook lands in the first 3 s, no cut falls inside a line, "
                   "and every timestamp exists in the script.")
    rec = (qa.get("recall") or {}).get("recall")
    if rec:
        st.markdown("**Did it find the moments a human picked?** (reference picks for this show)")
        st.table(pd.DataFrame([{"Within": k, "Reference moments found": v["all_refs"],
                                "Top-tier found": v["tier_A"]} for k, v in rec.items()]))
    if t and "skipped" not in t:
        st.markdown(f"**Head-to-head judge check:** the clip shown first won "
                    f"{t['position_bias_A_win_rate']:.0%} of {t['games']} games (fair is about 50%), "
                    f"and {t['swap_recheck']} verdicts held when the two clips were swapped.")
    qa_md = out / "qa_report.md"
    if qa_md.exists():
        with st.expander("Full quality report"):
            st.markdown(qa_md.read_text(encoding="utf-8"))
    log = out / "llm_log.txt"
    if log.exists():
        with st.expander("What the model got wrong, and how code fixed it"):
            st.text(log.read_text(encoding="utf-8")[:20000])

with tab_how:
    st.markdown("""
**Code does everything exact; the model only judges taste.**

1. **Parse** (code): every line of every script becomes a beat with its exact time, speaker and emotion tags.
2. **Cut points** (code): clips start on a camera shot and never cut in the middle of a line.
3. **Rule scores** (code): hook strength in the first 3 s, cliffhanger at the cut, makes sense alone.
4. **Model suggests and scores** (AI): it reads clips blind, like a stranger scrolling. It picks *line numbers*, never timestamps; code checks and repairs every suggestion.
5. **Head-to-head** (AI + code): the top clips play "which would you post, A or B?" rounds, which separates clips the 0–10 scores can't.
6. **Rank, posting plan, quality checks** (code).
""")

st.subheader("Download")
d1, d2, d3 = st.columns(3)
d1.download_button("promos.csv", (out / "promos.csv").read_bytes(), "promos.csv", width="stretch")
d2.download_button("promos.json", (out / "promos.json").read_bytes(), "promos.json", width="stretch")
if (out / "promos.md").exists():
    d3.download_button("Editor report (promos.md)", (out / "promos.md").read_bytes(), "promos.md", width="stretch")
