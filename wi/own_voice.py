"""Score the team's own-voice recordings with the frozen shipped pipeline (same code path as app.py).

Record in the demo with "Save these recordings for testing" ticked and the language typed in
(e.g. "Spanish", "English", "French"). Each press of Identify saves files named
<time>_<language>_<answer>.wav in data/own_voice/; files from one press form one utterance, exactly
as the app combined them. Nothing here is trained or tuned: models must match models/FROZEN.json.

Output: results/own_voice.md
Run: python -m wi.own_voice   (or: make ownvoice)
"""
import collections
import re

import numpy as np
import soundfile as sf

from .config import MODELS, NAMES, RESULTS, SR
from .decide import clip_logprobs, decide, useful, windows
from .validate import load_frozen

OWN = MODELS.parent / "data" / "own_voice"
TAGS = {"spanish": "spa", "espanol": "spa", "espaol": "spa", "english": "eng", "mam": "mam", "kiche": "quc",
        "quiche": "quc", "qeqchi": "kek", "kekchi": "kek", "kaqchikel": "cak", "qanjobal": "kjb"}


def describe(d):
    if d["kind"] == "named":
        return "English: no interpreter" if d["lang"] == "eng" else f"{NAMES[d['lang']]} interpreter"
    if d["kind"] == "mayan_try_two":
        return f"Mayan: try {NAMES[d['lang'][0]]}, then {NAMES[d['lang'][1]]}"
    return {"not supported": "Not a supported language", "not sure": "Not sure"}[d["kind"]]


def main():
    _, router, osm = load_frozen()              # refuses to run if the shipped models changed
    from .embed import Embedder
    from .phones import PhoneRecognizer
    groups = collections.defaultdict(list)
    for f in sorted(OWN.glob("*.wav")):
        m = re.match(r"(\d{8}-\d{6})_([a-z0-9]+)_(\d)\.wav$", f.name)
        if m:
            groups[(m[1], m[2])].append(f)
    if not groups:
        raise SystemExit(f"No own-voice recordings in {OWN}. Record in the demo with 'Save' ticked.")
    emb, rec = Embedder(), PhoneRecognizer()
    rows = []
    for (stamp, tag), files in groups.items():
        chunks = []
        for f in files:
            x, sr = sf.read(f, dtype="float32")
            if len(x) >= SR:
                chunks += windows(x)
        if not chunks:
            continue
        H, _ = emb(chunks)
        L7, L8 = clip_logprobs(H, rec(chunks), router, osm)
        d = decide(L7, L8, router, osm)
        true = TAGS.get(tag, tag)
        rows.append({"time": stamp, "language": NAMES.get(true, tag.capitalize()), "true": true,
                     "answers": len(files), "decision": describe(d), "useful": useful(d, true)})
    by = collections.defaultdict(list)
    for r in rows:
        by[r["language"]].append(r)
    out = ["# Own-voice test (team members, recorded with consent)\n",
           "Frozen shipped models, same decision rule as the demo, scored once; none of this audio was used for "
           "training or tuning. 'Useful' = the right interpreter, a 'try two' suggestion containing it, or (for a "
           "language the tool doesn't support) declining.\n",
           "| language | recordings | useful | what it said |", "|---|---|---|---|"]
    for lang, rs in sorted(by.items()):
        u = np.mean([r["useful"] for r in rs])
        out.append(f"| {lang} | {len(rs)} | {sum(r['useful'] for r in rs)}/{len(rs)} ({u:.0%}) | "
                   f"{collections.Counter(r['decision'] for r in rs).most_common()} |")
    out += ["\n## Every recording\n", "| time | spoken | answers | our answer | useful? |", "|---|---|---|---|---|"]
    out += [f"| {r['time']} | {r['language']} | {r['answers']} | {r['decision']} | {'yes' if r['useful'] else '**no**'} |"
            for r in rows]
    (RESULTS / "own_voice.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
