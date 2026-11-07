"""Rule-based signals: how intense is a beat, how good a hook / cliffhanger / standalone
window is.  These are cheap, deterministic and explainable.  They do three jobs:

1. generate and pre-rank candidates without any LLM (the "rules" engine),
2. act as hard validators on anything an LLM proposes,
3. give a reviewer a reason ("hook: slap SFX at 0.3s") they can check in seconds.

All weights live in this file so they can be tuned per genre.
"""
from __future__ import annotations

import math
import re
from .parser import Beat

# ---- emotion tags written by the screenwriter -------------------------------------------
TAG_WEIGHTS = {
    # high heat
    "screaming": 1.0, "shouting": 0.85, "furious": 0.9, "outraged": 0.85, "enraged": 0.9,
    "hysterical": 0.9, "shrieking": 0.9, "roaring": 0.9, "booming": 0.75, "thunderous": 0.8,
    "venomous": 0.8, "vicious": 0.8, "cruel": 0.75, "brutal": 0.8, "sadistic": 0.9,
    "gloating": 0.75, "smug": 0.6, "mocking": 0.6, "taunting": 0.65, "spiteful": 0.65,
    "threatening": 0.8, "menacing": 0.8, "deadly": 0.85, "lethally": 0.85, "dangerous": 0.75,
    "sinister": 0.7, "malicious": 0.7, "vindictive": 0.7, "chilling": 0.75, "murderous": 0.95,
    "shocked": 0.8, "devastated": 0.8, "horrified": 0.8, "terrified": 0.8, "panicked": 0.8,
    "agonized": 0.85, "sobbing": 0.7, "weeping": 0.6, "heartbroken": 0.65, "desperate": 0.7,
    "begging": 0.65, "pleading": 0.55, "anguished": 0.75, "shattered": 0.7, "crying": 0.6,
    "accusing": 0.6, "accusatory": 0.6, "defiant": 0.55, "fierce": 0.6, "scathing": 0.65,
    "piercing": 0.6, "betrayed": 0.65, "livid": 0.85, "disgusted": 0.6, "scornful": 0.6,
    "condemning": 0.65, "imperious": 0.55, "commanding": 0.5, "demanding": 0.5, "urgent": 0.55,
    "frantic": 0.75, "breathless": 0.5, "trembling": 0.45, "incredulous": 0.5, "revelatory": 0.85,
    "cold": 0.45, "steely": 0.45, "stern": 0.4, "harsh": 0.5, "callous": 0.55, "derisive": 0.6,
    "condescending": 0.5, "dismissive": 0.45, "unhinged": 0.9, "denial": 0.6, "realizing": 0.6,
    # cool / warm (lower: still useful inside a promo, weak as hook or cliff)
    "gentle": 0.15, "soft": 0.15, "calm": 0.15, "tender": 0.2, "warm": 0.15, "loving": 0.2,
    "reflective": 0.1, "playful": 0.15, "peaceful": 0.05, "content": 0.05, "amused": 0.1,
    "reassuring": 0.15, "quiet": 0.2, "sincere": 0.25, "earnest": 0.25, "solemn": 0.3,
}

# ---- words that make a scrolling viewer stop --------------------------------------------
SHOCK_WORDS = {
    # betrayal / romance stakes
    "marry": 0.6, "marrying": 0.7, "married": 0.5, "wedding": 0.4, "bride": 0.4, "husband": 0.35,
    "wife": 0.35, "betray": 0.7, "betrayed": 0.6, "lied": 0.5, "liar": 0.6, "love": 0.25,
    "divorce": 0.7, "pregnant": 0.9, "affair": 0.7, "mistress": 0.7, "disowned": 0.8,
    "sister": 0.25, "father": 0.2, "mother": 0.2, "son": 0.2, "daughter": 0.2,
    # danger / punishment
    "kill": 0.8, "killed": 0.8, "die": 0.6, "died": 0.6, "dead": 0.7, "death": 0.6, "hang": 0.8,
    "hangs": 0.8, "gallows": 0.85, "execute": 0.85, "execution": 0.85, "poison": 0.85,
    "poisoned": 0.85, "poisoning": 0.85, "arsenic": 0.8, "witch": 0.8, "witchcraft": 0.8,
    "burn": 0.6, "blood": 0.6, "bleed": 0.6, "crawl": 0.7, "arrest": 0.65, "thief": 0.6,
    "stole": 0.55, "steal": 0.5, "guards": 0.45, "dungeon": 0.5, "chains": 0.5, "tortured": 0.8,
    # status / reveal
    "duke": 0.6, "duchess": 0.5, "imposter": 0.7, "rightful": 0.55, "who are you": 0.7,
    "it's you": 0.7, "it was always me": 0.8, "invalid": 0.6, "not your": 0.4, "worthless": 0.6,
    "beggar": 0.45, "nobody": 0.4, "plain": 0.3, "servant": 0.35, "maid": 0.3,
}
SHOCK_RE = {w: re.compile(r"\b" + re.escape(w) + r"\b", re.I) for w in SHOCK_WORDS}

PHYSICAL = {
    "slap": 1.0, "slaps": 1.0, "strikes": 0.9, "strike": 0.8, "punch": 0.85, "whip": 0.9,
    "crop": 0.6, "crack": 0.5, "explod": 1.0, "explosion": 1.0, "detonation": 1.0, "fireball": 1.0,
    "burst": 0.7, "bursts": 0.7, "crash": 0.6, "shatter": 0.7, "drags": 0.6, "drag": 0.5,
    "grabs": 0.5, "seize": 0.6, "flings": 0.8, "throws": 0.7, "hurls": 0.7, "one knee": 0.8,
    "kiss": 0.6, "kisses": 0.6, "blood": 0.7, "bloody": 0.7, "bleeding": 0.7, "fire": 0.6,
    "blaze": 0.7, "flames": 0.6, "shackle": 0.5, "chained": 0.5, "collapses": 0.7, "faints": 0.7,
    "gasp": 0.35, "stunned": 0.4, "tackl": 0.7, "lunges": 0.6, "red-hot": 0.9, "iron": 0.2,
    "knife": 0.8, "dagger": 0.8, "sword": 0.7, "pistol": 0.8, "gun": 0.8, "noose": 0.9,
    "thorn": 0.5, "lacerat": 0.7, "gash": 0.7, "pounding": 0.5, "slam": 0.5, "kicked open": 0.7,
}
PHYSICAL_RE = {w: re.compile(r"\b" + re.escape(w), re.I) for w in PHYSICAL}

CLIFF_SFX = re.compile(r"cliffhanger|sting|drone|ominous|abrupt|cut to black|freez|swell|"
                       r"silence|hold(s|ing)? on|pounding|bursts? open|doors? (swing|burst)", re.I)
RESOLVING_SFX = re.compile(r"fad(e|es|ing) out|gentle|warm|tender|peaceful|soft", re.I)
CONTINUATION_START = re.compile(r"^(\.\.\.|…|and\b|but\b|then\b|so\b|or\b|because\b)", re.I)
REFERENCE_WORDS = re.compile(r"\b(again|last time|remember|as I said|like before|that night|"
                             r"you promised|back then|that day|what you did)\b", re.I)


def tag_heat(b: Beat) -> float:
    if not b.tags:
        return 0.3
    vals = []
    for t in b.tags:
        words = re.findall(r"[a-z\-]+", t.lower())
        w = max((TAG_WEIGHTS.get(x, 0.0) for x in words), default=0.0)
        if w == 0.0 and words:
            w = 0.3  # unknown tag: neutral
        vals.append(w)
    return max(vals)


def shock(text: str) -> tuple[float, list[str]]:
    hits = [(w, v) for w, v in SHOCK_WORDS.items() if SHOCK_RE[w].search(text)]
    if not hits:
        return 0.0, []
    vals = sorted((v for _, v in hits), reverse=True)
    return min(1.0, vals[0] + 0.25 * sum(vals[1:3])), [w for w, _ in hits]


def physical(text: str) -> tuple[float, list[str]]:
    hits = [(w, v) for w, v in PHYSICAL.items() if PHYSICAL_RE[w].search(text)]
    if not hits:
        return 0.0, []
    vals = sorted((v for _, v in hits), reverse=True)
    return min(1.0, vals[0] + 0.2 * sum(vals[1:2])), [w for w, _ in hits]


def beat_intensity(b: Beat) -> float:
    """0..1: how much this single beat grabs attention."""
    if b.kind == "dialogue":
        s, _ = shock(b.text)
        punct = 0.15 if "!" in b.text else 0.0
        q = 0.08 if b.text.rstrip().endswith("?") else 0.0
        short = 0.1 if len(b.text.split()) <= 6 else 0.0   # punchy lines land in 3 seconds
        return min(1.0, 0.5 * tag_heat(b) + 0.35 * s + punct + q + short)
    if b.kind in ("action", "sfx"):
        p, hits = physical(b.text)
        # a sound on its own is weaker than something the viewer sees happen to a person;
        # "a wet slap as the book hits the puddle" is not a slap (a real false positive we hit)
        if b.kind == "sfx":
            if hits and re.search(r"\b(book|puddle|water|rain|wave|door|paper|parchment)\b", b.text, re.I):
                p *= 0.5
            return min(1.0, 0.7 * p)
        return min(1.0, 0.85 * p)
    # camera: extreme close-ups / crash zooms carry a little energy, otherwise neutral
    if re.search(r"extreme close|crash zoom|whip pan|snap zoom", b.text, re.I):
        return 0.25
    return 0.0


def hook_strength(beats: list[Beat], t0: float, window: float = 3.0) -> tuple[float, Beat | None, list[str]]:
    """Best attention-grabber that STARTS inside the first `window` seconds."""
    best, best_b, notes = 0.0, None, []
    for b in beats:
        if b.kind == "camera" or b.start > t0 + window:
            continue
        v = beat_intensity(b)
        if b.kind == "dialogue":
            if CONTINUATION_START.match(b.text.strip()):
                v -= 0.35          # "...when she finally" makes no sense cold
            if b.start - t0 > 1.5:
                v -= 0.1           # still inside 3 s, but dead air first
        if v > best:
            best, best_b = v, b
    if best_b is None:
        notes.append("no dialogue or action in the first 3s (dead air)")
        return 0.0, None, notes
    if best_b.kind == "dialogue" and CONTINUATION_START.match(best_b.text.strip()):
        notes.append("hook line is a continuation of an earlier sentence")
    return max(0.0, min(1.0, best)), best_b, notes


def cliff_strength(ep_beats: list[Beat], included: list[Beat], t1: float,
                   is_episode_end: bool) -> tuple[float, Beat | None, list[str]]:
    """How unresolved is the moment we cut on?"""
    notes: list[str] = []
    content = [b for b in included if b.kind != "camera"]
    if not content:
        return 0.0, None, ["nothing happens at the end"]
    last_lines = [b for b in content if b.kind == "dialogue"][-2:]
    last = content[-1]
    score = 0.0

    # 1) the screenwriter already marks episode-ending cliffhangers
    tail_sfx = [b for b in included if b.kind == "sfx" and b.start >= t1 - 6]
    if any(CLIFF_SFX.search(b.text) for b in tail_sfx):
        score += 0.3
        notes.append("script marks a cliffhanger/sting here")
    if any(RESOLVING_SFX.search(b.text) for b in tail_sfx):
        score -= 0.2
        notes.append("music resolves (fade-out) at the cut")
    if is_episode_end:
        score += 0.2

    # 2) the last line(s): questions, threats, reveals and interruptions stay open
    for b in last_lines:
        txt = b.text.strip()
        w = 0.5 if b is last_lines[-1] else 0.25
        if txt.endswith("?") or txt.endswith("?!"):
            score += 0.35 * w / 0.5
        if txt.endswith(("...", "…", "—", "-")):
            score += 0.25 * w / 0.5
        s, _ = shock(txt)
        score += 0.4 * s * w / 0.5
        score += 0.3 * tag_heat(b) * w / 0.5
    if last.kind in ("action", "sfx"):
        p, _ = physical(last.text)
        score += 0.35 * p
        if re.search(r"burst|bursts|open|appear|arriv|enter|turns?|freez|realiz|smirk", last.text, re.I):
            score += 0.15   # someone just walked in / a look lands = open question

    # 3) what comes right after the cut: if it is the payoff, cutting here is a cliffhanger.
    after = [b for b in ep_beats if t1 - 0.01 <= b.start < t1 + 6 and b.kind != "camera"]
    if after:
        nxt = max(beat_intensity(b) for b in after)
        score += 0.2 * nxt
    # 4) tender resolution lines kill a cliffhanger
    if last_lines and tag_heat(last_lines[-1]) <= 0.2 and not last_lines[-1].text.strip().endswith("?"):
        score -= 0.15
        notes.append("ends on a soft/resolved line")
    # the signals above add up past 1.0 easily; squash so strong and very strong stay apart
    squashed = 1.0 - math.exp(-1.3 * max(0.0, score))
    return round(squashed, 4), (last_lines[-1] if last_lines else last), notes


UNIVERSAL = re.compile(r"\b(wedding|bride|husband|wife|sister|brother|mother|father|son|daughter|"
                       r"marry|marrying|married|divorce|baby|pregnant|boss|ceo|billionaire|duke|duchess|"
                       r"king|queen|prince|princess|maid|servant|poison|love|betray|cheat|kill|dead)\b", re.I)
# Capitalised words that are NOT names-of-people-on-screen are usually lore (places, houses,
# institutions). Generic for any show: no show-specific word list.
COMMON_CAPS = {
    "I", "I'm", "I'll", "I've", "I'd", "God", "Lord", "Lady", "Sir", "Madam", "Miss", "Mister", "Mr", "Mrs",
    "Ms", "Mother", "Father", "Mom", "Dad", "Your", "Grace", "Majesty", "Highness", "Duke", "Duchess",
    "King", "Queen", "Prince", "Princess", "Oh", "OK", "Okay", "No", "Yes", "Please", "Dear", "Heaven",
    "Christmas", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Doctor",
    "Captain", "Officer", "Ma'am", "Mommy", "Daddy", "Grandma", "Grandpa", "Uncle", "Aunt", "Sister", "Brother",
    # capitalised function words (sentence starts the regex can't see, e.g. after a quote or "...")
    "Her", "His", "Him", "She", "He", "The", "This", "That", "These", "Those", "They", "Them", "Their",
    "You", "Your", "We", "Our", "It", "Its", "And", "But", "Then", "So", "If", "When", "What", "Why",
    "How", "Who", "Where", "Not", "Now", "Let", "Look", "Stop", "Come", "Go", "Get", "Take", "Give",
    "Every", "All", "Some", "There", "Here", "Just", "Only", "Even", "Still", "Perhaps", "Maybe", "Never",
    "Good", "Well", "Enough", "Wait", "Help", "Thank", "Sorry", "Hold", "Leave", "Do", "Don't", "Did",
}
CAP_WORD = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,}(?:'s)?)\b")


def lore_terms(lines: list[Beat]) -> set[str]:
    """Proper nouns in the dialogue that aren't someone speaking in this clip."""
    speakers = set()
    for b in lines:
        speakers |= {w.title() for w in re.findall(r"[A-Z]{3,}", b.speaker)}
    found = set()
    for b in lines:
        for m in CAP_WORD.finditer(b.text.strip()):
            w = m.group(1).removesuffix("'s")
            if w in COMMON_CAPS or w in speakers:
                continue
            found.add(w)
    return found


def standalone_strength(included: list[Beat]) -> tuple[float, list[str]]:
    """Would someone who never saw the show follow this?  (Rules can only approximate.)"""
    notes: list[str] = []
    lines = [b for b in included if b.kind == "dialogue"]
    if not lines:
        return 0.3, ["no dialogue: relies on visuals only"]
    text = " ".join(b.text for b in lines)
    score = 0.75
    speakers = {b.speaker for b in lines}
    if len(speakers) > 4:
        score -= 0.06 * (len(speakers) - 4)
        notes.append(f"{len(speakers)} speakers")
    lore = lore_terms(lines)
    if lore:
        # a name or two is fine (the viewer learns them); a pile of houses/places is not
        score -= min(0.3, 0.05 * len(lore) + (0.05 if len(lore) >= 3 else 0.0))
        notes.append("unexplained names: " + ", ".join(sorted(lore)))
    refs = REFERENCE_WORDS.findall(text)
    if refs:
        score -= min(0.2, 0.07 * len(refs))
        notes.append("refers back to earlier events")
    if CONTINUATION_START.match(lines[0].text.strip()):
        score -= 0.15
    uni = {m.group(0).lower() for m in UNIVERSAL.finditer(text)}
    score += min(0.25, 0.06 * len(uni))   # universal stakes explain themselves
    offscreen = sum(1 for b in lines if b.offscreen) / len(lines)
    if offscreen > 0.5:
        score -= 0.1
        notes.append("mostly off-screen voices")
    return max(0.0, min(1.0, score)), notes


def density(included: list[Beat], t0: float, t1: float) -> float:
    """Share of the clip covered by dialogue/action (no dead air), plus mean intensity."""
    if t1 <= t0:
        return 0.0
    covered, cur_end = 0.0, t0
    for b in sorted((b for b in included if b.kind in ("dialogue", "action")), key=lambda x: x.start):
        s, e = max(b.start, cur_end), min(b.end, t1)
        if e > s:
            covered += e - s
            cur_end = e
    cov = covered / (t1 - t0)
    ints = [beat_intensity(b) for b in included if b.kind != "camera"]
    mean_i = sum(ints) / len(ints) if ints else 0.0
    return min(1.0, 0.5 * cov + 0.8 * mean_i)


def duration_fit(d: float) -> float:
    """1.0 inside the 25-45 s sweet spot for Reels/TikTok, tapering to the 20/60 s limits."""
    if d < 20 or d > 60:
        return 0.0
    if 25 <= d <= 45:
        return 1.0
    if d < 25:
        return 0.6 + 0.4 * (d - 20) / 5
    return 1.0 - 0.4 * (d - 45) / 15
