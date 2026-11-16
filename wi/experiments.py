"""All evaluations for the write-up, plus the final model.

E1  Bloom, held-out books (StratifiedGroupKFold by book) vs a leaky random clip split
E2  Cross-corpus: train on Bloom, test on corpora never seen in training
    (Tzijonel/Kaqchikel, COERLL/K'iche', Tachiwin/Mam, FLEURS/Spanish), clean + noise/reverb/phone
E3  Open set: does it say "I don't know" for languages it has no class for?
E4  Corpus-shortcut probe: can the same features tell Bloom Spanish from FLEURS Spanish?
Baselines: always-Spanish (status quo), MMS-LID zero-shot, MFCC + logistic regression.
Model variants:
    bloom            Bloom clips only
    bloom+aug        + one noise/reverb/phone-augmented copy of every training clip
    bloom+aug+fleurs + FLEURS-train Spanish (more Spanish speakers); makes FLEURS-test in-corpus

Outputs: results/metrics.json, results/report.md, results/*.png, results/errors.csv,
results/examples/ (wav + csv for the demo), models/router.joblib

Run: python -m wi.experiments
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CACHE, FEATS, LANGS, MANIFEST, MAYAN, MODELS, NAMES, RESULTS, SEED, SR, LID_LAYER

TARGET_ACC = 0.95       # abstention threshold is set so answered Bloom clips are >= 95% correct
VARIANTS = ["bloom", "bloom+aug", "bloom+aug+fleurs"]
FINAL = "bloom+aug+fleurs"
rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------ data
def load_view(view, man):
    d = np.load(FEATS / f"{view}.npz", allow_pickle=True)
    keep = ~pd.Series(d["clip_id"]).duplicated().values      # a few source URLs are listed twice
    d = {k: d[k][keep] for k in ["clip_id", "hidden", "mfcc", "zs_logits"]}
    df = pd.DataFrame({"clip_id": d["clip_id"]})
    df = df.merge(man, on="clip_id", how="left")
    return df, {"lid": d["hidden"], "mfcc": d["mfcc"], "zs": d["zs_logits"]}


def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))


def proba(clf, X):
    """Probabilities in LANGS order."""
    p = clf.predict_proba(X)
    order = [list(clf.classes_).index(l) for l in LANGS]
    return p[:, order]


# ------------------------------------------------------------------ metrics
def closed_metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    out = {"n": int(len(y)), "accuracy": float((y == pred).mean())}
    present = [l for l in LANGS if (y == l).any()]
    out["macro_f1"] = float(f1_score(y, pred, labels=present, average="macro")) if present else None
    out["recall"] = {l: float((pred[y == l] == l).mean()) for l in present}
    may = np.isin(y, MAYAN)
    if may.any():
        out["mayan_sent_to_spanish"] = float((pred[may] == "spa").mean())
    if (y == "spa").any():
        out["spanish_sent_to_mayan"] = float(np.isin(pred[y == "spa"], MAYAN).mean())
    yb, pb = np.where(may, "mayan", "spa"), np.where(np.isin(pred, MAYAN), "mayan", "spa")
    out["spanish_vs_mayan_accuracy"] = float((yb == pb).mean())
    return out


def with_abstain(y, P, tau):
    conf, pred = P.max(1), np.array(LANGS)[P.argmax(1)]
    ans = conf >= tau
    m = {"coverage": float(ans.mean())}
    if ans.any():
        m["accuracy_on_answered"] = float((np.asarray(y)[ans] == pred[ans]).mean())
        may = np.isin(y, MAYAN) & ans
        if may.any():
            m["mayan_sent_to_spanish_answered"] = float((pred[may] == "spa").mean())
    return m


def pick_tau_mayan(y, P):
    """Threshold on P(any Mayan) / P(Spanish) for the family-level answer: answered family
    decisions on held-out Bloom books must be >= 99% right."""
    y = np.asarray(y)
    pm = P[:, [LANGS.index(l) for l in MAYAN]].sum(1)
    fam_conf, fam_pred = np.maximum(pm, 1 - pm), np.where(pm >= 0.5, "mayan", "spa")
    fam_true = np.where(np.isin(y, MAYAN), "mayan", "spa")
    for t in np.round(np.arange(0.5, 0.995, 0.01), 2):
        ans = fam_conf >= t
        if ans.any() and (fam_true[ans] == fam_pred[ans]).mean() >= 0.99:
            return float(t)
    return 0.99


def route(P, tau, tau_m):
    """Decision per clip: ('lang', code) | ('mayan', [top1, top2]) | ('unsure', None)."""
    out = []
    mi = [LANGS.index(l) for l in MAYAN]
    for p in P:
        pm = p[mi].sum()
        if p.max() >= tau:
            out.append(("lang", LANGS[int(p.argmax())]))
        elif pm >= tau_m:
            top = [MAYAN[i] for i in np.argsort(p[mi])[::-1][:2]]
            out.append(("mayan", top))
        else:
            out.append(("unsure", None))
    return out


def routing_metrics(y, P, tau, tau_m):
    y = np.asarray(y)
    d = route(P, tau, tau_m)
    kind = np.array([k for k, _ in d])
    m = {"named_language": float((kind == "lang").mean()),
         "said_mayan_try_two": float((kind == "mayan").mean()),
         "said_not_sure": float((kind == "unsure").mean())}
    named = [(t, v) for t, (k, v) in zip(y, d) if k == "lang"]
    if named:
        m["named_language_correct"] = float(np.mean([t == v for t, v in named]))
    fam = [(t, v) for t, (k, v) in zip(y, d) if k == "mayan"]
    if fam:
        m["try_two_family_correct"] = float(np.mean([t in MAYAN for t, _ in fam]))
        m["try_two_contains_true"] = float(np.mean([t in v for t, v in fam]))
    if np.isin(y, MAYAN).any():
        mm = np.isin(y, MAYAN)
        m["mayan_routed_to_spanish"] = float(np.mean([k == "lang" and v == "spa"
                                                      for (k, v), t in zip(d, mm) if t]))
        # useful outcome: a correct named language, or a Mayan "try these two" that includes it
        m["useful_answer"] = float(np.mean([(k == "lang" and v == t) or (k == "mayan" and t in v)
                                            for (k, v), t in zip(d, y) if t in MAYAN]))
    return m


def group_bootstrap_f1(y, pred, groups, n=1000):
    y, pred, groups = map(np.asarray, (y, pred, groups))
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        vals.append(f1_score(y[s], pred[s], labels=LANGS, average="macro"))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def pick_tau(y, P):
    conf, pred = P.max(1), np.array(LANGS)[P.argmax(1)]
    for tau in np.round(np.arange(0.30, 0.995, 0.01), 2):
        ans = conf >= tau
        if ans.mean() > 0 and (np.asarray(y)[ans] == pred[ans]).mean() >= TARGET_ACC:
            return float(tau)
    return 0.99


# ------------------------------------------------------------------ training sets per variant
def training_set(variant, clean, F, aug, FA, idx_bloom_train):
    """Return (X, y, row_source) for a variant, restricted to the Bloom rows in idx_bloom_train."""
    keep_ids = set(clean.clip_id.values[idx_bloom_train])
    Xs, ys = [F["lid"][idx_bloom_train]], [clean.label.values[idx_bloom_train]]
    if "aug" in variant:
        m = aug.clip_id.isin(keep_ids).values
        Xs.append(FA["lid"][m]); ys.append(aug.label.values[m])
    if "fleurs" in variant:
        m = (clean.role == "train_extra").values
        Xs.append(F["lid"][m]); ys.append(clean.label.values[m])
        ma = (aug.role == "train_extra").values
        Xs.append(FA["lid"][ma]); ys.append(aug.label.values[ma])
    return np.concatenate(Xs), np.concatenate(ys)


# ------------------------------------------------------------------ main
def main():
    RESULTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    man = pd.read_parquet(MANIFEST)
    clean, F = load_view("clean", man)
    aug, FA = load_view("aug", man)
    R = {"layer": LID_LAYER, "target_accuracy_for_threshold": TARGET_ACC}

    btr = np.where((clean.role == "train") & (clean.corpus == "bloom"))[0]
    yb, gb = clean.label.values[btr], clean.group.values[btr]

    # ---------------- E1: Bloom CV
    sgk = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    folds = list(sgk.split(btr, yb, gb))
    E1, oof = {}, {}
    for variant in VARIANTS:
        P = np.zeros((len(btr), len(LANGS)))
        for tr, te in folds:
            X, y = training_set(variant, clean, F, aug, FA, btr[tr])
            P[te] = proba(make_clf().fit(X, y), F["lid"][btr[te]])
        oof[variant] = P
        pred = np.array(LANGS)[P.argmax(1)]
        E1[variant] = closed_metrics(yb, pred)
        E1[variant]["macro_f1_95ci_by_book"] = group_bootstrap_f1(yb, pred, gb)
        print("E1", variant, round(E1[variant]["macro_f1"], 3))
    # baselines on the same folds
    Pm = np.zeros((len(btr), len(LANGS)))
    for tr, te in folds:
        Pm[te] = proba(make_clf().fit(F["mfcc"][btr[tr]], yb[tr]), F["mfcc"][btr[te]])
    E1["baseline_mfcc"] = closed_metrics(yb, np.array(LANGS)[Pm.argmax(1)])
    E1["baseline_mms_lid_zero_shot"] = closed_metrics(yb, np.array(LANGS)[F["zs"][btr].argmax(1)])
    E1["baseline_always_spanish"] = closed_metrics(yb, np.array(["spa"] * len(btr)))
    # leakage: random clip split
    Pr = np.zeros((len(btr), len(LANGS)))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(btr, yb):
        Pr[te] = proba(make_clf().fit(F["lid"][btr[tr]], yb[tr]), F["lid"][btr[te]])
    E1["leaky_random_split"] = closed_metrics(yb, np.array(LANGS)[Pr.argmax(1)])
    E1["leakage_gap_macro_f1"] = E1["leaky_random_split"]["macro_f1"] - E1["bloom"]["macro_f1"]
    R["E1_bloom_heldout_books"] = E1

    tau = pick_tau(yb, oof[FINAL])
    tau_m = pick_tau_mayan(yb, oof[FINAL])
    R["abstain_threshold"] = tau
    R["family_threshold"] = tau_m
    R["E1_abstention_at_threshold"] = with_abstain(yb, oof[FINAL], tau)
    R["E1_routing"] = routing_metrics(yb, oof[FINAL], tau, tau_m)

    # ---------------- E2: cross-corpus, clean and degraded
    models = {}
    for variant in VARIANTS:
        X, y = training_set(variant, clean, F, aug, FA, btr)
        models[variant] = make_clf().fit(X, y)
    mfcc_model = make_clf().fit(F["mfcc"][btr], yb)
    E2 = {}
    views = {"clean": (clean, F)}
    for v in ["test_noise", "test_reverb", "test_phone"]:
        if (FEATS / f"{v}.npz").exists():
            views[v] = load_view(v, man)
    xc_rows = []
    for vname, (df, FF) in views.items():
        m = (df.role == "xcorpus").values
        sub = df[m]
        for (corpus, lab), g in sub.groupby(["corpus", "label"]):
            ii = np.where(m & (df.corpus == corpus).values & (df.label == lab).values)[0]
            key = f"{vname}|{corpus}|{lab}"
            E2[key] = {}
            for variant, clf in models.items():
                P = proba(clf, FF["lid"][ii])
                pred = np.array(LANGS)[P.argmax(1)]
                E2[key][variant] = {**closed_metrics(df.label.values[ii], pred),
                                    **with_abstain(df.label.values[ii], P, tau),
                                    "routing": routing_metrics(df.label.values[ii], P, tau, tau_m)}
                if vname == "clean" and variant == FINAL:
                    for j, k in enumerate(ii):
                        xc_rows.append(dict(clip_id=df.clip_id.values[k], corpus=corpus, true=lab,
                                            pred=pred[j], conf=float(P[j].max()), view=vname))
            if vname == "clean":
                E2[key]["baseline_mfcc"] = closed_metrics(
                    df.label.values[ii], np.array(LANGS)[proba(mfcc_model, FF["mfcc"][ii]).argmax(1)])
                E2[key]["baseline_mms_lid_zero_shot"] = closed_metrics(
                    df.label.values[ii], np.array(LANGS)[FF["zs"][ii].argmax(1)])
                E2[key]["baseline_always_spanish"] = closed_metrics(
                    df.label.values[ii], np.array(["spa"] * len(ii)))
                E2[key]["note"] = ("in-corpus for bloom+aug+fleurs (FLEURS-train used)"
                                   if corpus == "fleurs" else "")
    R["E2_cross_corpus"] = E2

    # ---------------- E5: how much real-world data closes the cross-corpus gap?
    # The model we ship is the k="all" one: Bloom + augmentation + FLEURS + half of every new
    # corpus. The other half of each new corpus stays held out for the demo examples.
    R["E5_adaptation"], final, test_groups = adaptation(clean, F, aug, FA, btr, tau, tau_m)

    # ---------------- E3: open set (shipped model)
    om = (clean.role == "openset").values
    Po = proba(final, F["lid"][om])
    odf = clean[om].assign(conf=Po.max(1), pred=np.array(LANGS)[Po.argmax(1)])
    E3 = {}
    for (corpus, lab), g in odf.groupby(["corpus", "label"]):
        Pg = Po[odf.index.get_indexer(g.index)]
        kinds = [k for k, _ in route(Pg, tau, tau_m)]
        E3[f"{corpus}|{lab}"] = {
            "n": int(len(g)), "abstained": float((g.conf < tau).mean()),
            "routing": {k: float(np.mean([x == k for x in kinds])) for k in ["lang", "mayan", "unsure"]},
            "when_answering_predicted": g[g.conf >= tau].pred.value_counts().to_dict()}
    # in-set = held-out Bloom books + held-out recordings of the new corpora (never trained on)
    xm = ((clean.role == "xcorpus") & (clean.corpus != "fleurs")
          & clean.group.isin(test_groups)).values
    in_conf = np.concatenate([oof[FINAL].max(1), proba(final, F["lid"][xm]).max(1)])
    E3["auroc_maxprob_inset_vs_openset"] = float(roc_auc_score(
        np.r_[np.ones(len(in_conf)), np.zeros(len(odf))], np.r_[in_conf, odf.conf.values]))
    E3["overall_abstain_rate_openset"] = float((odf.conf < tau).mean())
    R["E3_open_set"] = E3

    # ---------------- E4: corpus-shortcut probe (Spanish only)
    sm = (clean.label == "spa").values & clean.corpus.isin(["bloom", "fleurs"]).values
    yc = (clean.corpus.values[sm] == "fleurs").astype(int)
    Pc = np.zeros(sm.sum())
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(
            F["lid"][sm], yc, clean.group.values[sm]):
        c = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        Pc[te] = c.fit(F["lid"][sm][tr], yc[tr]).predict_proba(F["lid"][sm][te])[:, 1]
    R["E4_corpus_id_auroc_spanish_bloom_vs_fleurs"] = float(roc_auc_score(yc, Pc))

    # ---------------- held-out English books (English is a class: "no interpreter needed")
    em = (clean.role == "heldout_eng").values
    if em.any():
        Pe = proba(final, F["lid"][em])
        R["E3_english_heldout_books"] = {**closed_metrics(clean.label.values[em],
                                                          np.array(LANGS)[Pe.argmax(1)]),
                                         "routing": routing_metrics(clean.label.values[em], Pe, tau, tau_m),
                                         "sent_to_spanish": float((np.array(LANGS)[Pe.argmax(1)] == "spa").mean())}

    # ---------------- final model
    joblib.dump({"pipeline": final, "labels": LANGS, "tau": tau, "tau_mayan": tau_m,
                 "variant": FINAL, "layer": LID_LAYER}, MODELS / "router.joblib")

    # ---------------- errors and demo examples
    bl = clean.iloc[btr].assign(true=yb, pred=np.array(LANGS)[oof[FINAL].argmax(1)],
                                conf=oof[FINAL].max(1), view="bloom-heldout-books")
    hm = np.where(((clean.role == "xcorpus") & clean.group.isin(test_groups)).values | em)[0]
    Ph = proba(final, F["lid"][hm])
    held = clean.iloc[hm].assign(true=clean.label.values[hm], pred=np.array(LANGS)[Ph.argmax(1)],
                                 conf=Ph.max(1), view="new-corpus-heldout-recordings")
    errs = pd.concat([bl[["clip_id", "corpus", "true", "pred", "conf", "view"]],
                      held[["clip_id", "corpus", "true", "pred", "conf", "view"]]])
    pd.DataFrame(xc_rows).to_csv(RESULTS / "cross_corpus_predictions_bloom_only.csv", index=False)
    errs = errs.merge(man[["clip_id", "group", "src", "start", "note"]], on="clip_id")
    errs[errs.true != errs.pred].sort_values("conf", ascending=False).to_csv(
        RESULTS / "errors.csv", index=False)
    export_examples(errs, odf, tau)

    (RESULTS / "metrics.json").write_text(json.dumps(R, indent=1))
    figures(yb, oof[FINAL], odf, tau, E2)
    report(R)
    print("done; threshold", tau)


XCORPORA = ["coerll", "tachiwin", "tzijonel", "fleurs", "google_es", "librispeech"]   # last two: v2 diversity


def split_new_corpora(clean):
    """Split each new corpus 50/50 by recording/episode/lesson (deterministic).
    Returns ({corpus: [adaptation-pool groups]}, set(test groups))."""
    rs = np.random.default_rng(SEED)
    xm = (clean.role == "xcorpus").values
    pools, test_groups = {}, set()
    for c in XCORPORA:
        g = rs.permutation(sorted(clean[xm & (clean.corpus == c).values].group.unique()))
        pools[c] = list(g[: len(g) // 2]); test_groups |= set(g[len(g) // 2:])
    return pools, test_groups


def adaptation(clean, F, aug, FA, btr, tau, tau_m, ks=(0, 1, 2, 5, 10, "all"), reps=3):
    """Add k recordings per new corpus from the adaptation pool to training (Bloom + augmentation
    + FLEURS-train) and test on the untouched held-out recordings."""
    rs = np.random.default_rng(SEED + 1)
    xm = (clean.role == "xcorpus").values
    pools, test_groups = split_new_corpora(clean)
    te = np.where(xm & clean.group.isin(test_groups).values)[0]
    Xb, yb = training_set(FINAL, clean, F, aug, FA, btr)
    out, shipped = {}, None
    for k in ks:
        runs = []
        for r in range(1 if k in (0, "all") else reps):
            chosen = set()
            for c, p in pools.items():
                chosen |= set(p if k == "all" else list(rs.permutation(p))[:k])
            ad = np.where(xm & clean.group.isin(chosen).values)[0]
            clf = make_clf().fit(np.concatenate([Xb, F["lid"][ad]]),
                                 np.concatenate([yb, clean.label.values[ad]]))
            if k == "all":
                shipped = clf
            P = proba(clf, F["lid"][te])
            pred = np.array(LANGS)[P.argmax(1)]
            res = {}
            for c in XCORPORA:
                m = (clean.corpus.values[te] == c)
                yy = clean.label.values[te][m]
                res[c] = {"accuracy": float((pred[m] == yy).mean()),
                          **{kk: v for kk, v in routing_metrics(yy, P[m], tau, tau_m).items()
                             if kk in ("useful_answer", "mayan_routed_to_spanish", "named_language_correct")}}
            res["adapt_clips"] = int(len(ad))
            runs.append(res)
        out[str(k)] = {c: {kk: float(np.mean([r[c][kk] for r in runs if kk in r[c]]))
                           for kk in runs[0][c]} for c in XCORPORA}
        out[str(k)]["adapt_clips"] = float(np.mean([r["adapt_clips"] for r in runs]))
        print("E5 k=", k, {c: round(out[str(k)][c]["accuracy"], 3) for c in XCORPORA})
    out["test_clips"] = int(len(te))
    return out, shipped, test_groups


def export_examples(errs, odf, tau, per=4):
    out = RESULTS / "examples"; out.mkdir(exist_ok=True)
    picks = []
    for (c, t), g in errs.groupby(["corpus", "true"]):
        right = g[(g.true == g.pred) & (g.conf >= tau)].head(per)
        wrong = g[g.true != g.pred].head(2)
        picks += [right, wrong]
    for (c, t), g in odf.groupby(["corpus", "label"]):
        picks.append(g.assign(true=t).head(2)[["clip_id", "corpus", "true", "pred", "conf"]])
    ex = pd.concat(picks)
    ex["kind"] = np.where(ex.true.isin(LANGS), np.where(ex.true == ex.pred, "correct", "wrong"), "open-set")
    for cid in ex.clip_id:
        sf.write(out / f"{cid}.wav", np.load(CACHE / f"{cid}.npy").astype(np.float32), SR)
    ex.to_csv(out / "examples.csv", index=False)


def figures(yb, P, odf, tau, E2):
    pred = np.array(LANGS)[P.argmax(1)]
    cm = confusion_matrix(yb, pred, labels=LANGS, normalize="true")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    n = len(LANGS)
    ax.set_xticks(range(n), [NAMES[l] for l in LANGS], rotation=35, ha="right")
    ax.set_yticks(range(n), [NAMES[l] for l in LANGS])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Bloom, held-out books (row-normalised)")
    fig.tight_layout(); fig.savefig(RESULTS / "confusion_bloom.png", dpi=160); plt.close(fig)

    conf = P.max(1); correct = pred == yb
    taus = np.linspace(0, 0.99, 100)
    cov = [(conf >= t).mean() for t in taus]
    acc = [correct[conf >= t].mean() if (conf >= t).any() else np.nan for t in taus]
    rej = [(odf.conf < t).mean() for t in taus]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(taus, cov, label="target-language clips answered")
    ax.plot(taus, acc, label="accuracy of answered clips")
    ax.plot(taus, rej, label="other-language clips declined")
    ax.axvline(tau, color="gray", ls="--", lw=1)
    ax.set_xlabel("confidence threshold"); ax.legend(fontsize=8); ax.set_ylim(0, 1.02)
    ax.set_title("When should it say \"I don't know\"?")
    fig.tight_layout(); fig.savefig(RESULTS / "abstention.png", dpi=160); plt.close(fig)

    keys = sorted({k.split("|", 1)[1] for k in E2})
    vnames = ["clean", "test_noise", "test_reverb", "test_phone"]
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.2
    for i, v in enumerate(vnames):
        vals = [E2.get(f"{v}|{k}", {}).get(FINAL, {}).get("accuracy", np.nan) for k in keys]
        ax.bar(np.arange(len(keys)) + i * w, vals, w, label=v.replace("test_", ""))
    ax.set_xticks(np.arange(len(keys)) + 1.5 * w, [k.replace("|", "\n") for k in keys], fontsize=8)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title("Corpora never seen in training")
    fig.tight_layout(); fig.savefig(RESULTS / "cross_corpus.png", dpi=160); plt.close(fig)


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def report(R):
    E1 = R["E1_bloom_heldout_books"]
    L = ["# Results\n", "## E1 Bloom-Speech, held-out books\n",
         "| model | macro-F1 | accuracy | Mayan sent to Spanish | Spanish sent to Mayan |",
         "|---|---|---|---|---|"]
    for k in ["baseline_always_spanish", "baseline_mms_lid_zero_shot", "baseline_mfcc", *VARIANTS,
              "leaky_random_split"]:
        m = E1[k]
        L.append(f"| {k} | {m['macro_f1']:.3f} | {pct(m['accuracy'])} | "
                 f"{pct(m.get('mayan_sent_to_spanish'))} | {pct(m.get('spanish_sent_to_mayan'))} |")
    L += ["", f"Leakage gap (random split minus held-out books): {E1['leakage_gap_macro_f1']:.3f}",
          f"Book-bootstrap 95% CI for {FINAL}: {E1[FINAL]['macro_f1_95ci_by_book']}",
          f"Per-language recall ({FINAL}): " + ", ".join(
              f"{NAMES[l]} {pct(v)}" for l, v in E1[FINAL]['recall'].items()),
          f"Abstention threshold {R['abstain_threshold']}: {R['E1_abstention_at_threshold']}", "",
          "## E2 Corpora never seen in training\n",
          "| condition | corpus | language | n | always-Spanish | MMS zero-shot | MFCC | bloom | bloom+aug | bloom+aug+fleurs | answered at threshold |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in R["E2_cross_corpus"].items():
        cond, corpus, lab = k.split("|")
        b = lambda name: pct(v.get(name, {}).get("accuracy")) if name in v else ""
        L.append(f"| {cond} | {corpus} | {NAMES.get(lab, lab)} | {v[FINAL]['n']} | "
                 f"{b('baseline_always_spanish')} | {b('baseline_mms_lid_zero_shot')} | "
                 f"{b('baseline_mfcc')} | {b('bloom')} | {b('bloom+aug')} | {b(FINAL)} | "
                 f"{pct(v[FINAL].get('coverage'))} |")
    L += ["", "(FLEURS Spanish is in-corpus for bloom+aug+fleurs, because FLEURS-train is used.)", "",
          "### Two-level decision (final model)\n",
          f"Name a language if confidence >= {R['abstain_threshold']}; otherwise say \"Mayan: try X, "
          f"then Y\" if P(Mayan) >= {R['family_threshold']}; otherwise \"not sure\".\n",
          "| condition | corpus | language | named | named correct | 'Mayan, try two' | contains true | not sure | Mayan routed to Spanish | useful answer |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in R["E2_cross_corpus"].items():
        cond, corpus, lab = k.split("|")
        r = v[FINAL]["routing"]
        L.append(f"| {cond} | {corpus} | {NAMES.get(lab, lab)} | {pct(r['named_language'])} | "
                 f"{pct(r.get('named_language_correct'))} | {pct(r['said_mayan_try_two'])} | "
                 f"{pct(r.get('try_two_contains_true'))} | {pct(r['said_not_sure'])} | "
                 f"{pct(r.get('mayan_routed_to_spanish'))} | {pct(r.get('useful_answer'))} |")
    L += ["", f"Held-out Bloom books: {R['E1_routing']}", "",
          "## E5 Adding real-world recordings from each new corpus\n",
          f"Each new corpus is split in half by recording; k recordings per corpus from one half are "
          f"added to training, the other half is the test ({R['E5_adaptation']['test_clips']} clips). "
          "Mean of 3 random draws for k = 1-10.\n",
          "| recordings added per corpus | clips added | " + " | ".join(
              f"{c} accuracy" for c in XCORPORA) + " | " + " | ".join(
              f"{c} useful" for c in XCORPORA[:3]) + " |",
          "|---" * (2 + len(XCORPORA) + 3) + "|"]
    for k, v in R["E5_adaptation"].items():
        if k == "test_clips":
            continue
        L.append(f"| {k} | {v['adapt_clips']:.0f} | " + " | ".join(
            pct(v[c]['accuracy']) for c in XCORPORA) + " | " + " | ".join(
            pct(v[c].get('useful_answer')) for c in XCORPORA[:3]) + " |")
    L += ["",
          "## E3 Languages it should decline\n", "| corpus | language | n | declined | predicted when it answered |",
          "|---|---|---|---|---|"]
    for k, v in R["E3_open_set"].items():
        if "|" in k:
            c, lab = k.split("|")
            L.append(f"| {c} | {NAMES.get(lab, lab)} | {v['n']} | {pct(v['abstained'])} | "
                     f"{v['when_answering_predicted']} |")
    if "E3_english_heldout_books" in R:
        e = R["E3_english_heldout_books"]
        L += ["", f"English, held-out books (English is a class): accuracy {pct(e['accuracy'])}, "
              f"sent to a Spanish interpreter {pct(e['sent_to_spanish'])}."]
    L += ["", f"AUROC of confidence, target vs other languages: {R['E3_open_set']['auroc_maxprob_inset_vs_openset']:.3f}",
          "", "## E4 Corpus shortcut probe\n",
          f"AUROC telling Bloom Spanish from FLEURS Spanish with the same features: "
          f"{R['E4_corpus_id_auroc_spanish_bloom_vs_fleurs']:.3f} (near 1.0 means recording "
          "conditions are easy to see, so cross-corpus tests are essential)."]
    (RESULTS / "report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
