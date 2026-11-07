# Promo Finder

Reads a microdrama's timestamped screenplays and returns **ranked 20-60 second promo clips** for
TikTok / Reels / Shorts. Each clip opens on a hook in its first 3 seconds and cuts on a cliffhanger.
For each promo it gives the episode and start/end timestamps of every segment, plus the hook, the
cliffhanger, a one-line pitch, scores, warning flags and the transcript. It also gives a **posting order**.

Built for the Kuku AI Generalist assignment (Problem 1).

## What "plus" adds

This is a separate experimental version. It merges two earlier, independent builds and adds one new idea.

| | where it came from | what it does |
|---|---|---|
| **Head-to-head tournament** | new | The LLM's 0-10 scores bunch up at the top (all top-20 "post", many 10/10), so the top of the list was decided by noise. The top ~30 now play blind A-vs-B games over 5 Swiss rounds, and Bradley-Terry strengths re-rank them. It also **measures the judge**: how often the clip shown first wins (position bias), and whether the verdict survives swapping A and B. |
| **Merge any slate** | combining both builds | `--merge other.json` imports another tool's promos. They're snapped to this tool's legal cut points and re-judged blind with the same prompts, then compete in the same pool. The QA report shows which slate found each top moment. |
| **Scene-jump check** | the other build's QA | A clip that jumps location mid-clip is flagged and loses 4% per jump. |
| **Public-safe JSON** | the other build's redaction | `promos.public.json` has no transcripts. It can still quote short lines, so share it privately. |

```bash
python -m promo_finder --provider gemini --merge data/external/other_slate.json --tournament 28 --rounds 5
```

## Run it (5 minutes)

```bash
pip install -r requirements.txt
```

Put a show's episode scripts (`.txt`, one per episode) in `data/input/`. The input pack is
confidential and is **not** in this repo.

```bash
# rules only: free, about 30 s, no key needed
python -m promo_finder --input data/input --provider none

# with an LLM (best): free key from https://aistudio.google.com/apikey, saved in .env as GEMINI_API_KEY=...
python -m promo_finder --input data/input --provider gemini

# UI: upload scripts or a .zip, pick a model, browse saved runs
streamlit run app.py
```

Every run writes a folder in `outputs/`:

| file | what |
|---|---|
| `promos.md` | editor report: posting order + top 30 with hook, cliffhanger, why, transcript |
| `promos.csv` | every promo as one row (opens in Sheets) |
| `promos.json` | everything, machine-readable |
| `qa_report.md` | validator results, recall vs reference picks, flags to review |
| `run_meta.json` | model, settings, LLM calls, tokens, runtime |
| `llm_log.txt` | every model suggestion that was rejected, and why |

## Swap the model

Change one flag or one line in `.env`:

| provider | key | default model |
|---|---|---|
| `none` | - | rules engine only |
| `gemini` | `GEMINI_API_KEY` | `gemma-4-26b-a4b-it` (free, high quota); or `--model gemini-3.5-flash` |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| `openrouter` | `OPENROUTER_API_KEY` | a free Llama 3.3 70B |
| `ollama` | - (local) | `qwen2.5:3b` |
| `openai` | `OPENAI_API_KEY` + `OPENAI_BASE_URL` | any OpenAI-compatible API, e.g. Kuku's gateway |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |

**Free-tier reality (checked Nov 2026):** on a free AI Studio key, Gemini Flash models allow only about 20 requests per day each, which is not enough for one 50-episode show (about 110 calls). Gemma 4 models on the same key have a much larger quota, so the free default is `gemma-4-26b-a4b-it` with thinking set to `minimal` (`PROMO_THINKING`): about 8 s per call instead of about 70 s, with the same picks in a side-by-side test. A paid key or Kuku's own API just needs `--model`.

`--model` overrides the model name. Responses are cached in `.cache/llm/`, so re-runs are free and repeatable.
Clips are judged 3 per call (`--judge-batch`), never two from the same episode in one call, so a full 50-episode show needs about 110 calls. If Gemini's
free daily quota runs out, the model is unknown (404) or it is overloaded (503) 4 times in a row, the tool moves to the next fallback model and records
the switch in `llm_log.txt` and in the engine name.

## A new show

The tool doesn't depend on this show: no character names or places are hard-coded. Drop a
new show's scripts in a folder and run it. Optional extras:

- `aliases.json` next to the scripts merges speaker names for the same person, e.g.
  `{"THE MASKED MAN": "JACK"}`. "LADY X" is merged with "X" automatically, but only when "X" also speaks as a bare name.
- Episode numbers come from the `EPISODE NUMBER:` header, or from the file name if it's missing.
- Timestamps may be `MM:SS.ss` or `H:MM:SS`. Block headers may be `**NAME (Tags):**` or plain `NAME (Tags):`.
- `eval/reference_picks.json` (format: `eval/reference_picks.example.json`) is used only when its fingerprint matches the show; otherwise recall is skipped and says why. The real picks for the assignment's show quote the confidential scripts, so they ship with the raw outputs instead of this repo.

## Tests

```bash
python -m pytest -q
```

15 tests on synthetic scripts: timestamp formats, every block parsed, cuts never inside a line,
promos always 20-60 s with the hook in the first 3 s, made-up line IDs from the model rejected,
broken model JSON survived, batch judging matched clip by clip, and a full end-to-end run.

## How it works

```
scripts ─► parse (code) ─► candidates (rules) ─► LLM proposes (optional) ─► LLM judges top-K (optional)
                                                                         └─► rank ─► validate ─► reports
```

1. **Parse** (`parser.py`): every CAMERA / ACTION / SOUND FX / dialogue block becomes a beat with exact
   start/end seconds, speaker and the screenwriter's emotion tags. Pure code, checked against a block
   count of the source (3,727 of 3,727 blocks).
2. **Cut points** (`candidates.py`): in-points are camera-shot starts; out-points hold to the end of the
   shot; no cut ever lands inside a line of dialogue.
3. **Rules score** (`features.py`): hook strength in the first 3 s (emotion tags, shock words, physical
   action), cliffhanger strength at the cut (open question, threat, scripted "cliffhanger" sting, how big
   the next beat is), standalone (lore terms, callbacks, speaker count, universal stakes), density and
   length fit. The rules engine finds the moments (high recall) and checks everything else.
4. **Montages**: 2-4 short moments from different episodes in story order (hook first, cliffhanger last), 30-60 s.
5. **LLM** (`judge.py`): (a) reads whole episodes and proposes clips; (b) judges each candidate
   **blind**, seeing only the clip, the way a stranger scrolling would. **The model never writes a
   timestamp.** It points at numbered lines (`L12`), code maps them to script times, and re-checks every
   rule. Invalid suggestions are dropped and logged in `llm_log.txt`.
6. **Rank** (`rank.py`): final = 0.35 × rules + 0.65 × LLM; an episode diversity penalty; a posting order
   that leads with early-episode promos (a new viewer lands on episode 1) and never posts two clips of
   the same scene back to back.
7. **QA** (`qa.py`): hard validators on every promo, plus recall against `eval/reference_picks.json`.

## Confidentiality

`data/input/`, `outputs/`, `eval/reference_picks.json`, `.env` and `.cache/` are git-ignored: they contain Kuku's scripts, text quoted from them, or your API key. Share raw outputs by private link.

## Repo layout

```
promo_finder/   parser, features, candidates, llm, judge, rank, qa, report, pipeline, __main__
app.py          Streamlit UI
eval/           reference picks for recall
outputs/        raw outputs of every run
```
