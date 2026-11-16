"""Build the model the demo ships: embedding router fused with the phone n-gram router (A4).

Training data: Bloom (+ augmentation for the embedding model) + FLEURS-train + the adaptation half
of every new corpus. The fusion weight and both confidence thresholds are chosen on held-out Bloom
books only (out-of-fold), never on the new-corpus test recordings.
Scores the shipped model on held-out recordings, open-set languages and held-out English books.

Outputs: models/router.joblib (overwrites the embedding-only router), results/shipped_model.md/.json
Run: python -m wi.ship   (after wi.experiments, wi.phones)
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .advanced import (decide, phone_clf, phone_proba, phone_vectorizer, pick_tau_branch,
                       score_openset, setup)
from .config import FEATS, LANGS, LID_LAYER, MODELS, NAMES, RESULTS
from .experiments import (FINAL, make_clf, pick_tau, pick_tau_mayan, proba, routing_metrics,
                          training_set)

L = np.array(LANGS)


def fuse(Pe, Pp, w):
    lg = w * np.log(Pe + 1e-9) + (1 - w) * np.log(Pp + 1e-9)
    Q = np.exp(lg - lg.max(1, keepdims=True))
    return Q / Q.sum(1, keepdims=True)


def main():
    man, clean, F, aug, FA, btr, yb, gb, folds, pools, test_groups = setup()
    T = clean.clip_id.map(pd.read_parquet(FEATS / "phones.parquet").drop_duplicates("clip_id").set_index("clip_id").phones).fillna("").values
    extra = np.where((clean.role == "train_extra").values)[0]

    # out-of-fold on held-out Bloom books: choose fusion weight and thresholds
    Pe, Pp = np.zeros((len(btr), len(LANGS))), np.zeros((len(btr), len(LANGS)))
    for tr, te in folds:
        X, y = training_set(FINAL, clean, F, aug, FA, btr[tr])
        Pe[te] = proba(make_clf().fit(X, y), F["lid"][btr[te]])
        tri = np.concatenate([btr[tr], extra])
        tri = tri[T[tri] != ""]            # no phones recognised -> not phone training data
        vec = phone_vectorizer().fit(T[tri])
        Pp[te] = phone_proba(vec, phone_clf().fit(vec.transform(T[tri]), clean.label.values[tri]), T[btr[te]])
        empty = te[T[btr[te]] == ""]
        Pp[empty] = Pe[empty]            # no phones -> embedding only
    grid = np.round(np.linspace(0, 1, 11), 1)
    w = float(max(grid, key=lambda g: f1_score(yb, L[fuse(Pe, Pp, g).argmax(1)], average="macro")))
    Poof = fuse(Pe, Pp, w)
    tau, tau_m, tau_b = pick_tau(yb, Poof), pick_tau_mayan(yb, Poof), pick_tau_branch(yb, Poof)

    # final models: + adaptation half of every new corpus
    xm = (clean.role == "xcorpus").values
    ad = np.where(xm & clean.group.isin(set().union(*map(set, pools.values()))).values)[0]
    X, y = training_set(FINAL, clean, F, aug, FA, btr)
    emb = make_clf().fit(np.concatenate([X, F["lid"][ad]]), np.concatenate([y, clean.label.values[ad]]))
    tri = np.concatenate([btr, extra, ad])
    tri = tri[T[tri] != ""]
    vec = phone_vectorizer().fit(T[tri])
    pclf = phone_clf().fit(vec.transform(T[tri]), clean.label.values[tri])
    joblib.dump({"pipeline": emb, "phone_vectorizer": vec, "phone_clf": pclf, "fusion_weight": w,
                 "labels": LANGS, "tau": tau, "tau_mayan": tau_m, "tau_branch": tau_b,
                 "variant": FINAL + "+adapted+phones", "layer": LID_LAYER}, MODELS / "router.joblib")

    def shipped_proba(idx):
        pe = proba(emb, F["lid"][idx])
        out = fuse(pe, phone_proba(vec, pclf, T[idx]), w)
        out[T[idx] == ""] = pe[T[idx] == ""]      # no phones -> embedding only
        return out

    R = {"fusion_weight_on_embedding": w, "tau": tau, "tau_mayan": tau_m, "tau_branch": tau_b,
         "bloom_heldout_books_macro_f1_fused": float(f1_score(yb, L[Poof.argmax(1)], average="macro")),
         "bloom_heldout_books_macro_f1_embedding": float(f1_score(yb, L[Pe.argmax(1)], average="macro")),
         "bloom_heldout_books_routing": routing_metrics(yb, Poof, tau, tau_m)}
    te = np.where(xm & clean.group.isin(test_groups).values)[0]
    Pt = shipped_proba(te)
    R["heldout_recordings"] = {}
    for c in ["coerll", "tachiwin", "tzijonel", "fleurs", "google_es", "librispeech"]:
        m = clean.corpus.values[te] == c
        yy = clean.label.values[te][m]
        R["heldout_recordings"][c] = {"n": int(m.sum()), "accuracy": float((L[Pt[m].argmax(1)] == yy).mean()),
                                      "routing": routing_metrics(yy, Pt[m], tau, tau_m)}
    om = (clean.role == "openset").values
    Po = shipped_proba(np.where(om)[0])
    labs = clean.label.values[om]
    R["openset"] = {NAMES.get(l, l): score_openset(l, decide(Po[labs == l], tau, tau_b, tau_m))
                    for l in pd.unique(labs)}
    em = np.where((clean.role == "heldout_eng").values)[0]
    Pen = shipped_proba(em)
    R["english_heldout_books"] = {"accuracy": float((L[Pen.argmax(1)] == "eng").mean()),
                                  "sent_to_spanish": float((L[Pen.argmax(1)] == "spa").mean())}
    (RESULTS / "shipped_model.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))

    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"
    out = ["# Shipped model (embedding + phones, adapted)\n",
           f"Fusion weight on the embedding model: {w} (chosen on held-out Bloom books). Thresholds: "
           f"language {tau}, branch {tau_b}, Mayan {tau_m}.",
           f"Held-out Bloom books macro-F1: embedding {R['bloom_heldout_books_macro_f1_embedding']:.3f} -> "
           f"fused {R['bloom_heldout_books_macro_f1_fused']:.3f}.\n",
           "## Held-out recordings of the new corpora\n",
           "| corpus | n | accuracy | named a language | named correctly | 'Mayan, try two' | contains true | not sure | Mayan sent to Spanish | useful answer |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in R["heldout_recordings"].items():
        r = v["routing"]
        out.append(f"| {c} | {v['n']} | {pct(v['accuracy'])} | {pct(r['named_language'])} | "
                   f"{pct(r.get('named_language_correct'))} | {pct(r['said_mayan_try_two'])} | "
                   f"{pct(r.get('try_two_contains_true'))} | {pct(r['said_not_sure'])} | "
                   f"{pct(r.get('mayan_routed_to_spanish'))} | {pct(r.get('useful_answer'))} |")
    out += ["", f"English, held-out books: {pct(R['english_heldout_books']['accuracy'])} correct, "
            f"{pct(R['english_heldout_books']['sent_to_spanish'])} sent to Spanish.", "",
            "## Languages it has no class for (router alone, before the \"not supported\" check; full app: validation.md)\n", "| language | nearest covered relative | outcomes |", "|---|---|---|"]
    for l, v in R["openset"].items():
        near = v.pop("nearest_covered_relatives")
        out.append(f"| {l} | {', '.join(NAMES[x] for x in near) if near else 'none close'} | {v} |")
    (RESULTS / "shipped_model.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
