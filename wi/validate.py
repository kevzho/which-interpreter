"""Validation of the SHIPPED demo pipeline (exactly wi.decide, as used by app.py).

Part 1 - noise check on the language-disjoint open-set test: cluster bootstrap resampling whole
         unknown LANGUAGES and known recordings/books (FLEURS as one cluster per language, since its
         groups are sentences, not speakers). 95% CIs + paired difference vs max-probability.
Part 2 - end-to-end on held-out internal data: the shipped models + decision rule on held-out
         new-corpus recordings, held-out English books and open-set test languages, for utterances of
         1, 2 and 3 clips from the same recording (disjoint sets).
Part 3 - frozen external validation: the models must match models/FROZEN.json (written by
         `make freeze` before this runs); scored once on role == "external" clips (VoxPopuli, CIEMPIESS,
         EdAcc, Lingua Libre, Wikitongues). Unknown languages are tagged by whether the shipped open-set
         model trained on them as OTHER, used them only for tuning, or never saw them.

Run: python -m wi.validate
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import FEATS, LANGS, MAYAN, MODELS, NAMES, RESULTS, SEED, SR
from .decide import clip_logprobs, decide, useful
from .experiments import load_view, split_new_corpora

REPS = 2000
rng = np.random.default_rng(SEED)
METHODS = ["maxprob", "oe", "mahalanobis", "knn", "oe+maha"]


# ------------------------------------------------------------------ bootstrap
def cluster_bootstrap(df, methods, known_cluster=None, unknown_cluster="label", reps=REPS):
    """AUROC per method + paired difference vs maxprob, resampling clusters on each side."""
    known_cluster = known_cluster or ("cluster" if "cluster" in df else "group")
    kn, un = df[df.known], df[~df.known]
    kc = dict(kn.groupby(known_cluster).indices.items())
    uc = dict(un.groupby(unknown_cluster).indices.items())
    kg, ug = list(kc), list(uc)
    point = {m: roc_auc_score(np.r_[np.ones(len(kn)), np.zeros(len(un))], np.r_[kn[m], un[m]]) for m in methods}
    draws = {m: [] for m in methods}
    for _ in range(reps):
        ki = np.concatenate([kc[g] for g in rng.choice(kg, len(kg))])
        ui = np.concatenate([uc[g] for g in rng.choice(ug, len(ug))])
        y = np.r_[np.ones(len(ki)), np.zeros(len(ui))]
        for m in methods:
            draws[m].append(roc_auc_score(y, np.r_[kn[m].values[ki], un[m].values[ui]]))
    out = {}
    for m in methods:
        d = np.array(draws[m])
        out[m] = {"auroc": float(point[m]), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        if m != "maxprob" and "maxprob" in draws:
            diff = d - np.array(draws["maxprob"])
            out[m]["diff_vs_maxprob"] = float(point[m] - point["maxprob"])
            out[m]["diff_ci95"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
            out[m]["p_not_better"] = float((diff <= 0).mean())
    return out, len(kg), len(ug)


def cluster_rate_ci(values, clusters, reps=REPS):
    """Mean of a 0/1 outcome with a 95% CI resampling clusters."""
    v, c = np.asarray(values, float), np.asarray(clusters)
    idx = dict(pd.Series(range(len(c))).groupby(c).indices.items())
    keys = list(idx)
    if len(keys) < 2:
        return float(v.mean()), [None, None]
    d = [v[np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])].mean() for _ in range(reps)]
    return float(v.mean()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


# ------------------------------------------------------------------ helpers
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_frozen():
    f = MODELS / "FROZEN.json"
    frozen = json.loads(f.read_text())
    for name in ["router.joblib", "openset.joblib"]:
        if sha(MODELS / name) != frozen[name]:
            raise SystemExit(f"{name} changed after it was frozen; refusing to score the external set")
    return frozen, joblib.load(MODELS / "router.joblib"), joblib.load(MODELS / "openset.joblib")


def utterances(df, n, seed=SEED):
    """Disjoint sets of n clips from the same cluster (recording/speaker)."""
    r = np.random.default_rng(seed)
    sets = []
    for g, idx in df.groupby("cluster").indices.items():
        idx = r.permutation(idx)
        sets += [idx[i:i + n] for i in range(0, len(idx) - n + 1, n)]
    return sets


def score(df, L7, L8, router, osm, n):
    """Decisions for utterances of n clips; returns a per-utterance frame."""
    rows = []
    for s in utterances(df, n):
        d = decide(L7[s], L8[s], router, osm)
        true = df.label.values[s[0]]
        rows.append({"label": true, "cluster": df.cluster.values[s[0]], "source": df.source.values[s[0]],
                     "kind": d["kind"], "lang": d["lang"], "useful": useful(d, true),
                     "known_score": d["known_score"], "maxprob": d["conf"]})
    return pd.DataFrame(rows)


def summarise(u, known):
    """Per (source, language) summary with cluster CIs."""
    out = []
    for (src, lab), g in u.groupby(["source", "label"]):
        rate, ci = cluster_rate_ci(g.useful, g.cluster)
        r = {"source": src, "language": NAMES.get(lab, lab), "utterances": len(g), "clusters": g.cluster.nunique(),
             "useful": rate, "useful_ci95": ci}
        if lab in LANGS:
            r["named_correctly"] = float(((g.kind == "named") & (g.lang == lab)).mean())
            r["not_supported"] = float((g.kind == "not supported").mean())
            if lab in MAYAN:
                r["sent_to_spanish"] = float(((g.kind == "named") & (g.lang == "spa")).mean())
        else:
            r["declined"] = float(g.kind.isin(["not supported", "not sure"]).mean())
            named = g[g.kind == "named"].lang.map(lambda x: NAMES.get(x, x)).value_counts()
            r["named_instead"] = named.to_dict()
            r["mayan_try_two"] = float((g.kind == "mayan_try_two").mean())
        out.append(r)
    return out


# ------------------------------------------------------------------ main
def main():
    R = {}
    frozen, router, osm = load_frozen()
    R["frozen"] = frozen

    # Part 1: bootstrap on the open-set protocol scores
    sc = pd.read_parquet(RESULTS / "openset_test_scores.parquet")
    gt = {"tzj", "poc", "qum", "quv", "jac", "cac"}
    R["protocol_all"], nk, nu = cluster_bootstrap(sc, METHODS)
    R["protocol_guatemalan_mayan"], _, _ = cluster_bootstrap(sc[sc.known | sc.label.isin(gt)], METHODS)
    R["protocol_other_languages"], _, _ = cluster_bootstrap(sc[sc.known | ~sc.label.isin(gt)], METHODS)
    R["protocol_clusters"] = {"known": nk, "unknown_languages": nu}
    print("part 1 done", flush=True)

    # shared inputs
    man = pd.read_parquet(MODELS.parent / "data" / "manifest.parquet")
    clean, F = load_view("clean", man)
    ph = pd.read_parquet(FEATS / "phones.parquet").drop_duplicates("clip_id").set_index("clip_id").phones
    T = clean.clip_id.map(ph).fillna("").values
    cache = MODELS.parent / "data" / "cache" / "wav"
    is_ext = (clean.role == "external").values
    dur = pd.Series(np.inf, index=clean.index)
    dur[is_ext] = [len(np.load(cache / f"{c}.npy")) / SR for c in clean.clip_id[is_ext]]

    def frame(mask, source_col="corpus"):
        idx = np.where(mask)[0]
        df = clean.iloc[idx].assign(source=clean[source_col].values[idx]).reset_index(drop=True)
        df["cluster"] = np.where(df.corpus.eq("fleurs"), "fleurs:" + df.label, df.group)
        L7, L8 = clip_logprobs(F["lid"][idx], T[idx], router, osm)
        return df, L7, L8, idx

    # Part 2: end-to-end on held-out internal data
    pools, test_groups = split_new_corpora(clean)
    held = ((clean.role == "xcorpus") & clean.group.isin(test_groups)) | (clean.role == "heldout_eng") | \
           (clean.os_split.fillna("") == "test")
    df, L7, L8, idx = frame(held.values)
    R["internal"] = {}
    for n in [1, 2, 3]:
        u = score(df, L7, L8, router, osm, n)
        R["internal"][n] = summarise(u, known=None)
    R["internal_empty_phones"] = int((T[idx] == "").sum())
    print("part 2 done", flush=True)

    # Part 3: frozen external validation
    ext_mask = (clean.role == "external").values & (dur.values >= 1.0)      # the demo refuses < 1 s
    expected = set(man[man.role == "external"].clip_id)
    have = set(clean.clip_id[clean.role == "external"])
    R["external_missing_features"] = sorted(expected - have)
    os_split = man[man.os_split.fillna("") != ""].groupby("label").os_split.first().to_dict()
    df, L7, L8, idx = frame(ext_mask)
    df["seen"] = df.label.map(lambda l: "supported language" if l in LANGS else
                              {"train": "trained as OTHER", "tune": "tuning only"}.get(os_split.get(l), "never seen"))
    R["external_empty_phones"] = int((T[idx] == "").sum())
    R["external_short_excluded"] = int(((clean.role == "external").values & (dur.values < 1.0)).sum())
    R["external"] = {}
    for n in [1, 3]:
        u = score(df, L7, L8, router, osm, n)
        u["seen"] = u.label.map(dict(zip(df.label, df.seen)))
        R["external"][n] = summarise(u, known=None)
        for s in R["external"][n]:
            s["seen"] = dict(zip(df.label.map(lambda l: NAMES.get(l, l)), df.seen)).get(s["language"])
    # AUROC known vs unknown on single clips, overall and for never-seen unknowns only
    one = score(df, L7, L8, router, osm, 1).assign(known=lambda d: d.label.isin(LANGS))
    one["seen"] = one.label.map(dict(zip(df.label, df.seen)))
    one = one.rename(columns={"known_score": "oe"})
    R["external_auroc_all"], ek, eu = cluster_bootstrap(one, ["maxprob", "oe"], known_cluster="cluster")
    sub = one[one.known | one.seen.eq("never seen")]
    R["external_auroc_never_seen"], _, eu2 = cluster_bootstrap(sub, ["maxprob", "oe"], known_cluster="cluster")
    vp = one[one.source.eq("voxpopuli")]      # same source on both sides: source cannot stand in for language
    R["external_auroc_voxpopuli_only"], _, _ = cluster_bootstrap(vp, ["maxprob", "oe"], known_cluster="cluster")
    vpn = vp[vp.known | vp.seen.eq("never seen")]
    R["external_auroc_voxpopuli_never_seen"], _, _ = cluster_bootstrap(vpn, ["maxprob", "oe"], known_cluster="cluster")
    R["external_clusters"] = {"known_speakers": ek, "unknown_languages": eu, "never_seen_languages": eu2}

    # Part 4: fresh blind v2 set (Spanish/English from sources and accents never used for anything)
    is_e2 = (clean.role == "external2").values
    if is_e2.any():
        d2 = np.array([len(np.load(cache / f"{c}.npy")) / SR for c in clean.clip_id[is_e2]])
        m2 = is_e2.copy()
        m2[np.where(is_e2)[0]] = d2 >= 1.0
        df2, L7b, L8b, _ = frame(m2)
        R["external2"] = {n: summarise(score(df2, L7b, L8b, router, osm, n), known=None) for n in [1, 3]}
        R["external2_missing_features"] = sorted(set(man[man.role == "external2"].clip_id) - set(clean.clip_id[is_e2]))
    (RESULTS / "validation.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=str))
    report(R)


def report(R):
    f3 = lambda x: f"{x:.3f}"
    ci = lambda c: "-" if c[0] is None else f"[{c[0]:.3f}, {c[1]:.3f}]"
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"
    pci = lambda c: "-" if c[0] is None else f"{100 * c[0]:.0f}-{100 * c[1]:.0f}%"
    o = ["# Validation of the shipped demo\n",
         f"Models scored: router.joblib `{R['frozen']['router.joblib'][:16]}`, openset.joblib "
         f"`{R['frozen']['openset.joblib'][:16]}` (SHA-256, frozen at {R['frozen']['frozen_at']} before external scoring).\n",
         "## 1. Is the open-set improvement more than noise?\n",
         f"Cluster bootstrap ({REPS} reps): {R['protocol_clusters']['unknown_languages']} unknown test languages, "
         f"{R['protocol_clusters']['known']} known clusters (recordings/books; FLEURS = one cluster per language).\n"]
    for name, key in [("All test languages", "protocol_all"), ("6 Guatemalan Mayan (close relatives)", "protocol_guatemalan_mayan"),
                      ("Other test languages", "protocol_other_languages")]:
        o += [f"### {name}\n", "| score | AUROC | 95% CI | diff vs max-prob | 95% CI of diff | P(not better) |", "|---|---|---|---|---|---|"]
        for m, v in R[key].items():
            o.append(f"| {m} | {f3(v['auroc'])} | {ci(v['ci95'])} | " +
                     (f"{v['diff_vs_maxprob']:+.3f} | {ci(v['diff_ci95'])} | {v['p_not_better']:.3f} |" if "diff_ci95" in v else "- | - | - |"))
        o.append("")
    o += ["## 2. Shipped pipeline end-to-end on held-out internal data\n",
          "Held-out halves of the new corpora (same series as the adaptation half, likely same narrators), held-out "
          "English books, and open-set TEST languages (never trained on by the shipped models). 'Useful' = named "
          "correctly or 'Mayan, try two' containing the language (supported languages); declined (others). "
          "95% CIs resample recordings/speakers.\n",
          "| source | language | clusters | useful, 1 clip | useful, 2 clips | useful, 3 clips |", "|---|---|---|---|---|---|"]
    by = {n: {(r["source"], r["language"]): r for r in R["internal"][n]} for n in [1, 2, 3]}
    for key, r in sorted(by[1].items(), key=lambda kv: (kv[1]["language"] not in [NAMES[l] for l in LANGS], kv[0])):
        cells = [f"{pct(by[n][key]['useful'])} ({pci(by[n][key]['useful_ci95'])})" if key in by[n] else "-" for n in [1, 2, 3]]
        o.append(f"| {key[0]} | {key[1]} | {r['clusters']} | " + " | ".join(cells) + " |")
    if "external2" in R:
        o += ["", "## 3. Fresh blind test (v2): Spanish/English sources and accents never used for anything\n",
              "Google crowdsourced Spanish from ARGENTINA (accent held out), Multilingual LibriSpeech Spanish "
              "(audiobooks), People's Speech English (archive.org). Scored once with the frozen models above.\n",
              "| source | language | speakers/recordings | useful (1 clip) | useful (3 clips) | named correctly (1 clip) | 'not supported' (1 clip) |",
              "|---|---|---|---|---|---|---|"]
        b1 = {(r["source"], r["language"]): r for r in R["external2"][1]}
        b3 = {(r["source"], r["language"]): r for r in R["external2"][3]}
        for key, r in sorted(b1.items()):
            r3 = b3.get(key)
            o.append(f"| {key[0]} | {key[1]} | {r['clusters']} | {pct(r['useful'])} ({pci(r['useful_ci95'])}) | "
                     f"{pct(r3['useful']) if r3 else '-'} | {pct(r.get('named_correctly'))} | {pct(r.get('not_supported'))} |")
    o += ["", f"Clips with no recognised phones (embedding-only): {R['internal_empty_phones']}.", "",
          "## 4. v1 external set (VoxPopuli, CIEMPIESS, EdAcc, Lingua Libre, Wikitongues)\n",
          "This set was scored blind ONCE with the v1 models (results/v1/validation.md: e.g. only 9-13% of its "
          "Spanish was answered, because the open-set gate rejected unseen Spanish sources). That failure "
          "motivated v2, so the numbers below are a SECOND LOOK, not a blind test. v2 used no clip from it.\n",
          "Single clips (first 8 s; clips under 1 s excluded because the demo refuses them: "
          f"{R['external_short_excluded']}). The MMS-LID backbone itself was trained on FLEURS and evaluated on "
          "VoxPopuli by Meta, so 'never used' refers to this project's fitting and tuning.\n",
          "| source | language | status | clusters | useful (1 clip) | useful (3 clips, same speaker) | named instead |",
          "|---|---|---|---|---|---|---|"]
    e1 = {(r["source"], r["language"]): r for r in R["external"][1]}
    e3 = {(r["source"], r["language"]): r for r in R["external"][3]}
    for key, r in sorted(e1.items(), key=lambda kv: ({"supported language": 0, "never seen": 1, "tuning only": 2,
                                                     "trained as OTHER": 3}.get(kv[1].get("seen"), 4), kv[0])):
        r3 = e3.get(key)
        o.append(f"| {key[0]} | {key[1]} | {r.get('seen')} | {r['clusters']} | {pct(r['useful'])} ({pci(r['useful_ci95'])}) | "
                 f"{pct(r3['useful']) if r3 else '-'} | {r.get('named_instead', '')} |")
    o += ["", f"External AUROC (known vs unknown, single clips; {R['external_clusters']['known_speakers']} known speakers):"]
    for name, key in [("all unknown languages", "external_auroc_all"), ("never-seen unknown languages only", "external_auroc_never_seen"),
                      ("VoxPopuli only (same source both sides)", "external_auroc_voxpopuli_only"),
                      ("VoxPopuli only, never-seen languages", "external_auroc_voxpopuli_never_seen")]:
        for m, v in R[key].items():
            o.append(f"- {name}, {m}: {f3(v['auroc'])} {ci(v['ci95'])}" +
                     (f"; diff vs max-prob {v['diff_vs_maxprob']:+.3f} {ci(v['diff_ci95'])}" if "diff_ci95" in v else ""))
    if R["external_missing_features"]:
        o.append(f"\nWARNING: {len(R['external_missing_features'])} external clips have no features.")
    (RESULTS / "validation.md").write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
