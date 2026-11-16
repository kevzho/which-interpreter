"""Synthetic-voice test: macOS text-to-speech voices read intake-style answers; the frozen shipped
pipeline (same code path as app.py) scores them once. Nothing is trained or tuned.

Why: a team without Spanish speakers can still check the Spanish path, and synthetic speech is a
recording condition none of our training audio has. Each voice reads three answers (name, town,
reason for visit); we score each answer alone and all three combined, as the app does.

Output: results/synthetic_voices.md
Run: python -m wi.synthetic_voices   (or: make synthetic; macOS only)
"""
import collections
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .config import LANGS, MANIFEST, NAMES, RESULTS
from .decide import clip_logprobs, decide, useful, windows
from .own_voice import describe
from .validate import load_frozen

TEXT = {
    "spa": ["Me llamo María López y tengo treinta y dos años.", "Vivo en Huehuetenango, cerca del mercado, con mi familia.",
            "Vengo porque me duele mucho el estómago desde hace tres días."],
    "eng": ["My name is Maria Lopez and I am thirty two years old.", "I live in Oakland, near the market, with my family.",
            "I came because my stomach has hurt a lot for three days."],
    "fra": ["Je m'appelle Marie Lopez et j'ai trente-deux ans.", "J'habite à Lyon, près du marché, avec ma famille.",
            "Je viens parce que j'ai très mal au ventre depuis trois jours."],
    "por": ["Meu nome é Maria Lopes e tenho trinta e dois anos.", "Moro em Recife, perto do mercado, com minha família.",
            "Vim porque meu estômago dói muito há três dias."],
    "cmn": ["我叫李玛丽，今年三十二岁。", "我住在奥克兰，离市场很近，和家人一起住。", "我来是因为我的肚子已经疼了三天了。"],
}
VOICES = {
    "spa": ["Paulina", "Mónica"] + [f"{n} (Spanish ({r}))" for n in ["Eddy", "Flo", "Grandma", "Grandpa", "Reed", "Rocko"]
                                    for r in ["Mexico", "Spain"]],
    "eng": ["Samantha"] + [f"{n} (English (US))" for n in ["Eddy", "Flo", "Grandma", "Grandpa", "Reed", "Rocko"]],
    "fra": ["Jacques"] + [f"{n} (French (France))" for n in ["Eddy", "Flo"]],
    "por": ["Luciana"] + [f"{n} (Portuguese (Brazil))" for n in ["Eddy", "Flo"]],
    "cmn": [f"{n} (Chinese (China mainland))" for n in ["Eddy", "Flo", "Reed"]],
}
SPLIT = {"train": "trained as 'other'", "tune": "used only to set the line", "test": "never heard"}


def speak(voice, text, path):
    try:
        r = subprocess.run(["say", "-v", voice, "-o", str(path), "--file-format=WAVE", "--data-format=LEI16@16000", text],
                           capture_output=True, timeout=60)
        return r.returncode == 0 and path.exists()
    except subprocess.TimeoutExpired:
        return False


def main():
    _, router, osm = load_frozen()              # refuses to run if the shipped models changed
    from .embed import Embedder
    from .phones import PhoneRecognizer
    emb, rec = Embedder(), PhoneRecognizer()
    man = pd.read_parquet(MANIFEST)
    seen = {l: SPLIT.get(sorted(man.os_split[man.label == l].dropna().replace("", np.nan).dropna().unique())[0], "?")
            for l in TEXT if l not in LANGS}
    rows, tmp = [], Path(tempfile.mkdtemp())
    for lang, voices in VOICES.items():
        for v in voices:
            files = [tmp / f"{lang}_{abs(hash(v))}_{i}.wav" for i in range(3)]
            if not all(speak(v, t, f) for t, f in zip(TEXT[lang], files)):
                print("skipped voice (not installed or hung):", v)
                continue
            clips = [windows(sf.read(f, dtype="float32")[0]) for f in files]
            for n, groups in ((1, [[c] for c in clips]), (3, [clips])):
                for g in groups:
                    chunks = [w for c in g for w in c]
                    H, _ = emb(chunks)
                    L7, L8 = clip_logprobs(H, rec(chunks), router, osm)
                    d = decide(L7, L8, router, osm)
                    rows.append({"lang": lang, "voice": v, "answers": n, "said": describe(d), "useful": useful(d, lang),
                                 "top_guess": NAMES[d["top"]], "kind": d["kind"]})
    df = pd.DataFrame(rows)
    out = ["# Synthetic-voice test (macOS text-to-speech)\n",
           "Frozen shipped models, same decision rule as the demo, scored once; nothing trained or tuned on these. "
           "Each voice reads three intake answers (name, town, reason for visit); we score each answer alone and all "
           "three combined. 'Useful' = the right interpreter (Spanish), 'no interpreter needed' (English), or declining "
           "(languages the tool doesn't support). Synthetic speech is a recording condition absent from our training data.\n",
           "| language | status | voices | useful, 1 answer | useful, 3 answers | what it said (1 answer) |", "|---|---|---|---|---|---|"]
    for lang in TEXT:
        s = df[df.lang == lang]
        if s.empty:
            continue
        one, three = s[s.answers == 1], s[s.answers == 3]
        status = "supported" if lang in LANGS else seen[lang]
        out.append(f"| {NAMES.get(lang, {"cmn": "Mandarin"}.get(lang, lang))} | {status} | {s.voice.nunique()} | {one.useful.sum()}/{len(one)} | "
                   f"{three.useful.sum()}/{len(three)} | {dict(collections.Counter(one.said).most_common())} |")
    sp = df[(df.lang == "spa") & (df.kind == "not supported")]
    if len(sp):
        out.append(f"\nSpanish declined by the \"not supported\" check: {len(sp)} runs; the language classifier underneath "
                   f"still had Spanish as its top guess in {(sp.top_guess == NAMES['spa']).sum()} of them. This is the same "
                   "failure as the blind tests: the check treats unfamiliar-sounding Spanish as another language.")
    out += ["\n## Every run\n", "| language | voice | answers | what it said | top guess underneath |", "|---|---|---|---|---|"]
    out += [f"| {NAMES.get(r.lang, {"cmn": "Mandarin"}.get(r.lang, r.lang))} | {r.voice} | {r.answers} | {r.said} | {r.top_guess} |" for r in df.itertuples()]
    (RESULTS / "synthetic_voices.md").write_text("\n".join(out) + "\n")
    print("\n".join(out[:12]))


if __name__ == "__main__":
    main()
