"""v1 vs v2 models on the same data, plus where rejected clips would have gone.

Both model versions are frozen (models/v1/ and models/); nothing is fitted here. This quantifies
(a) whether the v2 data changed anything on the fresh blind set, and (b) for clips the open-set gate
rejects, how often the 7-class language router underneath had the right answer.

Output: results/v1_vs_v2.md
Run: python -m wi.compare_versions
"""
import joblib
import numpy as np
import pandas as pd

from .config import CACHE, FEATS, LANGS, MODELS, NAMES, RESULTS, SR
from .decide import clip_logprobs, decide, useful
from .experiments import load_view

SETS = [("google_es_ar", "Spanish, Argentina (blind v2)"), ("mls", "Spanish, audiobooks (blind v2)"),
        ("peoples_speech", "English, archive.org (blind v2)"), ("ciempiess", "Spanish, radio (v1 external)"),
        ("voxpopuli:spa", "Spanish, parliament (v1 external)"), ("voxpopuli:eng", "English, parliament (v1 external)"),
        ("edacc", "English, conversation (v1 external)")]


def main():
    man = pd.read_parquet(MODELS.parent / "data" / "manifest.parquet")
    clean, F = load_view("clean", man)
    ph = pd.read_parquet(FEATS / "phones.parquet").drop_duplicates("clip_id").set_index("clip_id").phones
    T = clean.clip_id.map(ph).fillna("").values
    models = {"v1": (joblib.load(MODELS / "v1" / "router.joblib"), joblib.load(MODELS / "v1" / "openset.joblib")),
              "v2": (joblib.load(MODELS / "router.joblib"), joblib.load(MODELS / "openset.joblib"))}
    rows = []
    for key, name in SETS:
        corpus, _, lab = key.partition(":")
        m = (clean.corpus == corpus).values & clean.role.isin(["external", "external2"]).values
        if lab:
            m &= (clean.label == lab).values
        idx = np.where(m)[0]
        idx = idx[[len(np.load(CACHE / f"{c}.npy")) / SR >= 1.0 for c in clean.clip_id.values[idx]]]
        r = {"set": name, "clips": len(idx)}
        for v, (router, osm) in models.items():
            L7, L8 = clip_logprobs(F["lid"][idx], T[idx], router, osm)
            ds = [decide(L7[j:j + 1], L8[j:j + 1], router, osm) for j in range(len(idx))]
            y = clean.label.values[idx]
            rej = np.array([d["kind"] == "not supported" for d in ds])
            r[f"{v}_useful"] = np.mean([useful(d, t) for d, t in zip(ds, y)])
            r[f"{v}_rejected"] = rej.mean()
            top_ok = np.array([d["top"] == t for d, t in zip(ds, y)])
            r[f"{v}_router_right_when_rejected"] = top_ok[rej].mean() if rej.any() else None
        rows.append(r)
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"
    out = ["# v1 vs v2 models on the same clips\n",
           "Both versions frozen; single clips; clips under 1 s excluded. 'Router right when rejected' = among clips the "
           "'I don't know' gate rejected, how often the 7-class language router's top guess was the correct language.\n",
           "| clips | n | v1 useful | v2 useful | v1 rejected | v2 rejected | router right when rejected (v1 / v2) |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['set']} | {r['clips']} | {pct(r['v1_useful'])} | {pct(r['v2_useful'])} | {pct(r['v1_rejected'])} | "
                   f"{pct(r['v2_rejected'])} | {pct(r['v1_router_right_when_rejected'])} / {pct(r['v2_router_right_when_rejected'])} |")
    (RESULTS / "v1_vs_v2.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
