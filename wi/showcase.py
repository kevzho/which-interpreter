"""Side-by-side examples for the write-up: the shipped pipeline's answer next to the correct answer.

Picks are deterministic (first clip by clip_id with a correct answer and first with a wrong answer in
each category), so they are not cherry-picked. Categories cover held-out recordings, held-out English,
the blind v2 set, the v1 external set and languages the tool does not support.

Output: results/side_by_side.md
Run: python -m wi.showcase
"""
import numpy as np
import pandas as pd
import joblib

from .config import FEATS, LANGS, NAMES, RESULTS
from .decide import clip_logprobs, decide, useful
from .experiments import load_view, split_new_corpora

CATS = [
    ("Mam, held-out radio recording (Tachiwin)", lambda d, tg: (d.corpus == "tachiwin") & (d.label == "mam") & d.group.isin(tg)),
    ("K'iche', held-out course audio (COERLL)", lambda d, tg: (d.corpus == "coerll") & d.group.isin(tg)),
    ("Kaqchikel, held-out podcast episode (Tzijonel)", lambda d, tg: (d.corpus == "tzijonel") & d.group.isin(tg)),
    ("Spanish, blind accent (Argentina)", lambda d, tg: d.corpus.eq("google_es_ar")),
    ("Spanish, parliament (VoxPopuli, v1 external set)", lambda d, tg: d.corpus.eq("voxpopuli") & d.label.eq("spa")),
    ("English, held-out storybook", lambda d, tg: d.role.eq("heldout_eng")),
    ("Popti' (no class; nearest relative Q'anjob'al)", lambda d, tg: d.label.eq("jac") & d.os_split.eq("test")),
    ("Sakapulteko (no class; nearest relative K'iche')", lambda d, tg: d.label.eq("quv") & d.os_split.eq("test")),
    ("French (no class)", lambda d, tg: d.label.eq("fra") & d.os_split.eq("test")),
]


def answer(d):
    if d["kind"] == "named":
        return "English: no interpreter" if d["lang"] == "eng" else f"{NAMES[d['lang']]} interpreter"
    if d["kind"] == "mayan_try_two":
        return f"Mayan: try {NAMES[d['lang'][0]]}, then {NAMES[d['lang'][1]]}"
    return {"not supported": "Not a supported language", "not sure": "Not sure"}[d["kind"]]


def main():
    man = pd.read_parquet(RESULTS.parent / "data" / "manifest.parquet")
    clean, F = load_view("clean", man)
    clean["os_split"] = clean.os_split.fillna("")
    _, tg = split_new_corpora(clean)
    router, osm = joblib.load(RESULTS.parent / "models" / "router.joblib"), joblib.load(RESULTS.parent / "models" / "openset.joblib")
    ph = pd.read_parquet(FEATS / "phones.parquet").drop_duplicates("clip_id").set_index("clip_id").phones
    rows = []
    for name, sel in CATS:
        idx = np.where(sel(clean, tg).values)[0]
        idx = idx[np.argsort(clean.clip_id.values[idx])]
        H, T = F["lid"][idx], clean.clip_id.values[idx]
        T = pd.Series(T).map(ph).fillna("").values
        L7, L8 = clip_logprobs(H, T, router, osm)
        got = {}
        for j, i in enumerate(idx):
            d = decide(L7[j:j + 1], L8[j:j + 1], router, osm)
            ok = useful(d, clean.label.values[i])
            if ok not in got:
                got[ok] = (i, d, T[j])
            if len(got) == 2:
                break
        for ok in [True, False]:
            if ok in got:
                i, d, phones = got[ok]
                true = clean.label.values[i]
                correct = (("English: no interpreter" if true == "eng" else f"{NAMES[true]} interpreter")
                           if true in LANGS else "Not a supported language")
                rows.append(f"| {name} | {correct} | {answer(d)} | {'yes' if ok else '**no**'} | `{phones[:60]}` |")
    out = ["# Side-by-side examples (shipped pipeline, single clip)\n",
           "Deterministic picks: the first correct and the first wrong answer (by clip id) in each category.\n",
           "| clip | correct answer | our answer | right? | phones heard (start) |", "|---|---|---|---|---|"] + rows
    (RESULTS / "side_by_side.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
