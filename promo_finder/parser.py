"""Parse Kuku-style timestamped screenplays into a flat, time-ordered list of beats.

Every block in the script (CAMERA / ACTION / SOUND FX / a character line) becomes one
Beat with a start and end time in seconds, relative to the start of its episode.
Nothing here uses an LLM: parsing must be exact, because every timestamp we output
comes from this file and never from a model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

TIME_RE = re.compile(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)")
# "**MIA (Cold):**" (this pack) and also a plain "MIA (Cold):" line
BLOCK_HEAD_RE = re.compile(r"^(?:\*\*\s*([^*]+?)\s*:?\s*\*\*\s*:?|([A-Z][A-Z0-9 '.\-/&]*(?:\([^)]*\)\s*)*):)\s*$")
# "Start: (00:00.00) | End: (00:04.80)"  also "Start 0:00.0 - End 0:04.8", "start: 00:01 to end: 00:03"
START_END_RE = re.compile(
    r"start\s*:?\s*\(?\s*([\d:.]+)\s*\)?\s*(?:\||,|-|–|to)?\s*end\s*:?\s*\(?\s*([\d:.]+)\s*\)?", re.I
)


def _head(m: re.Match) -> str:
    return (m.group(1) or m.group(2) or "").strip()
NON_SPEAKER = {"CAMERA", "ACTION", "SOUND FX", "SFX", "MUSIC", "TEXT", "SUPER", "TITLE"}


def to_seconds(ts: str) -> float:
    """'01:39.10' -> 99.1 ; '1:02:03.5' -> 3723.5 (H:MM:SS also accepted)."""
    m = TIME_RE.search(ts)
    if not m:
        raise ValueError(f"bad timestamp: {ts!r}")
    h = int(m.group(1)) if m.group(1) else 0
    return h * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def fmt(sec: float) -> str:
    """Seconds -> MM:SS.ss, the same format the scripts use."""
    sec = max(0.0, sec)
    m = int(sec // 60)
    return f"{m:02d}:{sec - 60 * m:05.2f}"


@dataclass
class Beat:
    idx: int                 # position inside the episode (0-based, time-ordered)
    episode: int
    kind: str                # camera | action | sfx | dialogue
    start: float
    end: float
    text: str
    speaker: str = ""        # dialogue only, normalised (e.g. "MIA")
    tags: list[str] = field(default_factory=list)   # emotion tags, e.g. ["Cold", "Commanding"]
    offscreen: bool = False  # O.S. / V.O.
    scene: int = 0

    @property
    def dur(self) -> float:
        return self.end - self.start

    def label(self) -> str:
        if self.kind == "dialogue":
            t = f" ({', '.join(self.tags)})" if self.tags else ""
            return f'{self.speaker}{t}: "{self.text}"'
        return f"[{self.kind.upper()}] {self.text}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start_ts"], d["end_ts"] = fmt(self.start), fmt(self.end)
        return d


@dataclass
class Scene:
    number: int
    heading: str
    start: float
    end: float


@dataclass
class Episode:
    number: int
    logline: str
    continuity: str
    scenes: list[Scene]
    beats: list[Beat]
    source: str

    @property
    def duration(self) -> float:
        return max((b.end for b in self.beats), default=0.0)

    def window(self, start: float, end: float) -> list[Beat]:
        """Beats that start inside [start, end)."""
        return [b for b in self.beats if start - 1e-6 <= b.start < end - 1e-6]


# Generic, show-independent: "LADY MIA" and "MIA" are the same speaker.
# Show-specific aliases ("THE MASKED MAN" = "JACK") can be passed in via
# load_show(aliases=...) or an aliases.json next to the scripts; nothing is hard-coded.
HONORIFICS = re.compile(r"^(LADY|LORD|SIR|DAME|MISS|MRS|MR|MS|DR|FATHER|SISTER|PRINCE|PRINCESS|"
                        r"KING|QUEEN|DUKE|DUCHESS|COUNT|COUNTESS|CAPTAIN|THE)\s+(?=[A-Z])")
SPEAKER_ALIASES: dict[str, str] = {}


def normalise_speaker(name: str, aliases: dict[str, str] | None = None) -> str:
    n = re.sub(r"\s+", " ", name.upper()).strip()
    al = aliases if aliases is not None else SPEAKER_ALIASES
    return al.get(n, n)


def _merge_honorifics(eps: list["Episode"]) -> None:
    """'LADY MIA' -> 'MIA', but only when 'MIA' also speaks as a bare name somewhere.
    So 'LORD ASH' and 'LADY ASH' (two different people) stay apart."""
    names = {b.speaker for e in eps for b in e.beats if b.kind == "dialogue"}
    for e in eps:
        for b in e.beats:
            if b.kind != "dialogue":
                continue
            stripped = HONORIFICS.sub("", b.speaker)
            if stripped != b.speaker and stripped in names:
                b.speaker = stripped


def _parse_speaker(head: str) -> tuple[str, list[str], bool]:
    """'LADY MIA (Outraged, Accusing)' -> ('MIA', ['Outraged','Accusing'], False)."""
    parens = re.findall(r"\(([^)]*)\)", head)
    name = re.sub(r"\([^)]*\)", "", head).strip().rstrip(":").strip()
    offscreen = False
    tags: list[str] = []
    for p in parens:
        for t in (x.strip() for x in re.split(r"[,/]", p) if x.strip()):
            u = t.upper().replace(" ", "")
            if u in {"O.S.", "V.O.", "O.S", "V.O", "OS", "VO", "OFF-SCREEN", "OFFSCREEN", "CONT'D", "CONTINUES"}:
                offscreen = offscreen or u not in {"CONT'D", "CONTINUES"}
                continue
            tags.append(t)
    return normalise_speaker(name), tags, offscreen


def parse_episode(path: str | Path) -> Episode:
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()

    number, logline, continuity = 0, "", ""
    scenes: list[Scene] = []
    raw: list[dict] = []
    cur_scene = 0
    pending_scene: dict = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        low = line.lower()
        if low.startswith("episode number:"):
            m = re.search(r"\d+", line)
            number = int(m.group()) if m else 0
        elif low.startswith("episode logline:"):
            logline = line.split(":", 1)[1].strip()
        elif low.startswith("continuity notes:"):
            continuity = line.split(":", 1)[1].strip()
        elif re.match(r"^scene\s+\d+", low):
            cur_scene = int(re.search(r"\d+", line).group())
            pending_scene = {"number": cur_scene, "heading": "", "start": None, "end": None}
        elif low.startswith("scene heading:"):
            pending_scene["heading"] = line.split(":", 1)[1].strip()
        elif low.startswith("scene start:"):
            pending_scene["start"] = to_seconds(line.split(":", 1)[1])
        elif low.startswith("scene end:"):
            pending_scene["end"] = to_seconds(line.split(":", 1)[1])
            scenes.append(Scene(**pending_scene))
        else:
            m = BLOCK_HEAD_RE.match(line)
            if m and i + 1 < len(lines):
                head = _head(m)
                # find the Start/End line (normally the next line)
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                se = START_END_RE.search(lines[j]) if j < len(lines) else None
                if se:
                    k = j + 1
                    body: list[str] = []
                    while k < len(lines):
                        nxt = lines[k].strip()
                        if BLOCK_HEAD_RE.match(nxt) or re.match(r"^scene\s+\d+", nxt.lower()) or nxt.startswith("--- SCRIPT"):
                            break
                        if nxt:
                            body.append(nxt)
                        k += 1
                    text = " ".join(body).strip().strip('"').strip("“”")
                    raw.append({
                        "head": head, "start": to_seconds(se.group(1)), "end": to_seconds(se.group(2)),
                        "text": text, "scene": cur_scene,
                    })
                    i = k
                    continue
        i += 1

    beats: list[Beat] = []
    for r in raw:
        head_up = re.sub(r"\([^)]*\)", "", r["head"]).strip().rstrip(":").upper()
        if head_up in NON_SPEAKER:
            kind = {"SOUND FX": "sfx", "SFX": "sfx", "MUSIC": "sfx"}.get(head_up, head_up.lower())
            if kind not in {"camera", "action", "sfx"}:
                kind = "action"
            beats.append(Beat(0, number, kind, r["start"], r["end"], r["text"], scene=r["scene"]))
        else:
            name, tags, off = _parse_speaker(r["head"])
            beats.append(Beat(0, number, "dialogue", r["start"], r["end"], r["text"],
                              speaker=name, tags=tags, offscreen=off, scene=r["scene"]))
    # stable time order: start time, then camera first so a shot "contains" its content
    order = {"camera": 0, "action": 1, "sfx": 2, "dialogue": 3}
    beats.sort(key=lambda b: (b.start, order[b.kind]))
    for n, b in enumerate(beats):
        b.idx = n
    if number == 0:   # no "EPISODE NUMBER:" header: take the first number in the file name
        m = re.search(r"(\d+)", path.stem)
        number = int(m.group(1)) if m else 0
    for b in beats:
        b.episode = number
    return Episode(number, logline, continuity, scenes, beats, path.name)


def load_show(folder: str | Path, aliases: dict[str, str] | None = None) -> list[Episode]:
    """All episode scripts in a folder (recursively), sorted by episode number.

    Optional `aliases.json` in the folder ({"THE MASKED MAN": "JACK"}) merges
    speaker names that refer to the same person."""
    folder = Path(folder)
    al_file = folder / "aliases.json"
    if aliases is None and al_file.exists():
        import json
        aliases = {k.upper(): v.upper() for k, v in json.loads(al_file.read_text(encoding="utf-8")).items()}
    SPEAKER_ALIASES.clear()
    SPEAKER_ALIASES.update(aliases or {})
    files = sorted(p for p in folder.rglob("*.txt") if not p.name.startswith("._"))
    eps = []
    for p in files:
        try:
            e = parse_episode(p)
        except Exception as ex:  # one broken file must not kill the whole show
            print(f"warning: could not parse {p.name}: {ex}")
            continue
        if e.beats:
            eps.append(e)
    # renumber duplicates / zeros by file order so every episode has a unique number
    seen = set()
    for i, e in enumerate(sorted(eps, key=lambda e: (e.number or 10**6, e.source)), 1):
        if e.number in seen or e.number == 0:
            e.number = max(seen | {0}) + 1
            for b in e.beats:
                b.episode = e.number
        seen.add(e.number)
    eps.sort(key=lambda e: e.number)
    _merge_honorifics(eps)
    return eps


if __name__ == "__main__":
    import sys
    show = load_show(sys.argv[1] if len(sys.argv) > 1 else "data/input")
    for e in show:
        kinds = {}
        for b in e.beats:
            kinds[b.kind] = kinds.get(b.kind, 0) + 1
        print(f"Ep {e.number:>2}  {fmt(e.duration)}  scenes={len(e.scenes)}  {kinds}")
