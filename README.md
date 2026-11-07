# Promo Finder

Reads a microdrama's timestamped screenplays and returns **ranked 20-60 second promo clips** for
TikTok / Reels / Shorts. Each clip opens on a hook in its first 3 seconds and cuts on a cliffhanger.
For each promo it gives the episode and start/end timestamps of every segment, plus the hook, the
cliffhanger, a one-line pitch, scores, warning flags and the transcript. It also gives a **posting order**.

Built for the Kuku AI Generalist assignment (Problem 1: Find the Promos).

**Live app: https://promo-finder-yojit.streamlit.app** (upload a show's episode scripts, press Run; rules-only works with no key,
or paste your own API key in the sidebar for the AI judge). The full results on the assignment's input pack quote Kuku's
confidential scripts, so they were shared privately with the submission and are not in this repo.

## Run it (5 minutes)

Needs Python 3.10 or newer.

```bash
git clone https://github.com/yogo8347/promo-finder.git
cd promo-finder
pip install -r requirements.txt
```

Put a show's episode scripts (`.txt`, one per episode) in a folder, e.g. `data/input/`. The input pack is
confidential and is **not** in this repo.

### The app (easiest)

```bash
streamlit run app.py
```

On the hosted app, keys stay in your browser session only; nothing is saved on the server or shown to other visitors.

1. **Sidebar → Model:** pick a provider. For Gemini, open *API key*, paste a free key from
   https://aistudio.google.com/apikey and press **Save key** (it goes into a local `.env` that git ignores).
   **Test this model** makes one call to check it; **Show models my key can use** lists models without using quota.
   Choose `none` to run on rules only, with no key.
2. **Sidebar → Speed:** *Quick* (about 5-10 min, for a live demo) or *Full* (about 35 min on a free key, best results).
3. **1. Screenplays:** upload the `.txt` files or one `.zip`, or point at a folder. Optionally limit episodes (`1-5`).
4. **2. Find promos:** press **Run**. Or open a saved run from `outputs/`.
5. **3. Results:** the slate (with a line-by-line view of any promo), the posting plan, the quality checks, downloads.

### The command line

```bash
# rules only: free, about 30 s, no key, no internet
python -m promo_finder --input data/input --provider none

# full run with the LLM (key in .env as GEMINI_API_KEY=...)
python -m promo_finder --input data/input --provider gemini

# quick run for a live demo (about 5-10 min)
python -m promo_finder --input data/input --provider gemini --no-propose --judge-top 30 --tournament 16 --rounds 3
```

Every run writes a folder in `outputs/`:

| file | what |
|---|---|
| `promos.md` | editor report: posting order + top 30 with hook, cliffhanger, why, transcript |
| `promos.csv` | every promo as one row (opens in Sheets) |
| `promos.json` | everything, machine-readable, including each head-to-head game and its reason |
| `promos.public.json` | the same without transcripts (can still quote short lines: share privately) |
| `qa_report.md` | validator results, recall vs reference picks, judge checks, flags to review |
| `run_meta.json` | model, settings, LLM calls, tokens, runtime |
| `llm_log.txt` | every model suggestion that code repaired or rejected, and why |

## How it works

```
scripts ─► parse (code) ─► cut points (code) ─► candidates + rule scores (code)
       ─► LLM suggests clips by line ID + judges the top clips blind (AI; code checks every answer)
       ─► rank (code) ─► head-to-head A-vs-B on the top 28 (AI + code) ─► posting order ─► validate ─► reports
```

1. **Parse** (`parser.py`): every CAMERA / ACTION / SOUND FX / dialogue block becomes a beat with exact
   start/end seconds, speaker and the screenwriter's emotion tags. Pure code (3,727 of 3,727 blocks).
2. **Cut points** (`candidates.py`): in-points are camera-shot starts; out-points hold to the end of the
   shot; no cut ever lands inside a line of dialogue.
3. **Rule scores** (`features.py`): hook strength in the first 3 s (emotion tags, shock words, physical
   action), cliffhanger strength at the cut, makes-sense-alone (unexplained names, callbacks, universal stakes),
   pace and length. Montages: 2-4 moments from different episodes, in story order, 30-60 s.
4. **LLM** (`judge.py`): reads whole episodes and suggests clips, and judges candidates **blind** (only the clip,
   like a stranger scrolling). **The model never writes a timestamp.** It points at numbered lines (`L12`); code maps
   them to script time, repairs a wrong length to the nearest legal cut, and drops anything else (logged).
5. **Rank** (`rank.py`): 0.35 × rules + 0.65 × LLM, minus 4% per scene jump inside a clip, plus an episode-diversity penalty.
6. **Head-to-head** (`tournament.py`): 0-10 scores bunch up at the top, so the top 28 play blind
   "which would you post, A or B?" games over 5 Swiss rounds; Bradley-Terry strengths re-rank them (70/30 with the
   score). It also **measures the judge**: how often the clip shown first wins, and whether a verdict survives swapping A and B.
7. **Fact-check** (`grounding.py`): the AI's hook / cliffhanger text for the top 30 is checked against the script. Every quote must
   appear in the clip (code), and an AI fact-check flags events that are not in the clip (e.g. something that happens after the cut).
   Corrections are logged; if a fix still fails the code check, the real script line is used.
8. **Posting order**: early-episode promos first (a new viewer lands on episode 1), never the same episode twice in a row,
   never two posts that cut on the same moment.
9. **QA** (`qa.py`): hard validators on every promo, plus recall against `eval/reference_picks.json`.

## Swap the model

Change one flag, one line in `.env`, or the sidebar in the app:

| provider | key | default model |
|---|---|---|
| `none` | - | rules engine only |
| `gemini` | `GEMINI_API_KEY` | `gemma-4-26b-a4b-it` (free, high quota); or `--model gemini-3.5-flash` |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| `openrouter` | `OPENROUTER_API_KEY` | a free Llama 3.3 70B |
| `ollama` | - (local) | `qwen2.5:3b` |
| `openai` | `OPENAI_API_KEY` + `OPENAI_BASE_URL` | any OpenAI-compatible API, e.g. Kuku's gateway |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |

**Free-tier reality (checked Nov 2026):** Gemini Flash models allow only about 20 free requests a day each, not enough
for one show (a full run is about 200 calls, measured at 288k input / 53k output tokens). Gemma 4 on the same key has a much larger quota, so the free default is
`gemma-4-26b-a4b-it` with thinking set to `minimal` (`PROMO_THINKING`). Responses are cached in `.cache/llm/`, so re-runs
are free and identical. If the model is unknown (404), out of daily quota, or overloaded 4 times in a row, the tool moves
to a fallback model and records it in `llm_log.txt`.

## A new show

Nothing about this show is hard-coded. Drop a new show's scripts in a folder and run it.

- `aliases.json` next to the scripts merges speaker names for one person, e.g. `{"THE MASKED MAN": "JACK"}`.
  "LADY X" is merged with "X" automatically, but only when "X" also speaks as a bare name.
- Episode numbers come from the `EPISODE NUMBER:` header, or from the file name if it's missing.
- Timestamps may be `MM:SS.ss` or `H:MM:SS`. Block headers may be `**NAME (Tags):**` or plain `NAME (Tags):`.
- `eval/reference_picks.json` (format: `eval/reference_picks.example.json`) is used only when its fingerprint matches
  the show; otherwise recall is skipped and says why.
- `--merge other_slate.json` imports another tool's promos, snaps them to legal cuts, re-judges them blind and lets them
  compete; the QA report then shows which slate found each top moment.

## Measuring it

```bash
# exact token cost of a full run, replayed from the cache (no generation quota used)
python eval/measure_cost.py --input data/input

# is the head-to-head stable? re-run it with different random first-round pairings
python -m promo_finder --provider gemini --h2h-seed 1 --name stability_seed1

# a second, independent reference set: a different model picks promo moments blind, then score a run against it
python eval/independent_picks.py --input data/input
python eval/independent_picks.py --score outputs/<run>/promos.json
```

## Tests

```bash
python -m pytest -q
```

23 tests on synthetic scripts: timestamp formats, every block parsed, cuts never inside a line, promos always
20-60 s with the hook in the first 3 s, made-up line IDs rejected, broken model JSON survived, batch judging,
the head-to-head (Bradley-Terry, Swiss pairing, a position-biased fake judge), imported slates, scene jumps,
the fact-check (quotes not in the clip, fallback to the script line) and the no-repeated-ending posting rule.

## Confidentiality

`data/`, `outputs/` (except `.gitkeep`), `eval/reference_picks.json`, `.env` and `.cache/` are git-ignored: they hold
Kuku's scripts, text quoted from them, or your API key. Share raw outputs by private link.

## Repo layout

```
promo_finder/   parser, features, candidates, llm, judge, tournament, grounding, external, rank, qa, report, pipeline, __main__
app.py          Streamlit app
tests/          unit + end-to-end tests
eval/           reference-pick format (the real picks stay private), measure_cost.py, independent_picks.py
outputs/        runs are written here (git-ignored)
```
