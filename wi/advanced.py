"""Four extensions beyond the baseline router (see results/advanced.md).

A1  Several utterances from the same speaker: average evidence over 1, 2, 4, 8 clips of one
    recording ("ask the person to say three things").
A2  Linear concept erasure (LEACE, Belrose et al. 2023) of the *recording corpus*, estimated
    within each language, tested leave-one-corpus-out. Controls: no erasure, naive (not
    within-language) erasure, and erasing random directions of the same rank.
A3  Mayan family tree: tree distance of mistakes, a hierarchical branch -> language model, a
    branch-level fallback answer, open-set languages scored against their nearest covered
    relative, and split-conformal prediction sets (and where their guarantee breaks).
A4  Phonotactics: a universal phone recogniser's IPA output -> phone n-gram classifier, fused
    with the embedding model; top phone n-grams per language for interpretability.

Run: python -m wi.advanced   (after wi.experiments; A4 needs wi.phones)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATS, LANGS, MANIFEST, MAYAN, NAMES, RESULTS, SEED
from .experiments import (FINAL, XCORPORA, load_view, make_clf, pick_tau, pick_tau_mayan, proba,
                          routing_metrics, split_new_corpora, training_set)

L = np.array(LANGS)

# ------------------------------------------------------------------ Mayan family tree
# child -> parent. Branches follow the standard (Kaufman) classification of Mayan languages.
PARENT = {
    "mayan": "root", "nonmayan": "root", "spa": "nonmayan", "eng": "nonmayan",
    "eastern": "mayan", "western": "mayan", "yucatecan": "mayan", "huastecan": "mayan",
    "greater_kichean": "eastern", "mamean": "eastern",
    "kichean": "greater_kichean", "qeqchian": "greater_kichean", "poqom": "greater_kichean",
    "quc": "kichean", "cak": "kichean", "tzj": "kichean", "quv": "kichean", "qum": "kichean",
    "kek": "qeqchian", "poc": "poqom", "mam": "mamean",
    "qanjobalan_chujean": "western", "cholan_tseltalan": "western",
    "qanjobalan": "qanjobalan_chujean", "chujean": "qanjobalan_chujean",
    "kjb": "qanjobalan", "jac": "qanjobalan", "cac": "chujean", "toj": "chujean",
    "ctu": "cholan_tseltalan", "tzh": "cholan_tseltalan", "tzo": "cholan_tseltalan",
    "yua": "yucatecan", "itz": "yucatecan", "lac": "yucatecan", "hus": "huastecan",
}
# Branches the router can answer with when it cannot name the language.
BRANCH = {"spa": "Spanish", "eng": "English", "quc": "K'ichean", "cak": "K'ichean",
          "kek": "Q'eqchi'", "mam": "Mamean", "kjb": "Q'anjob'alan"}
BRANCHES = ["Spanish", "English", "K'ichean", "Q'eqchi'", "Mamean", "Q'anjob'alan"]


def ancestors(x):
    out = [x]
    while out[-1] != "root":
        out.append(PARENT[out[-1]])
    return out


def tree_dist(a, b):
    A, B = ancestors(a), ancestors(b)
    common = next(n for n in A if n in B)
    return A.index(common) + B.index(common)


def nearest_targets(lang):
    """Covered languages closest in the tree; None if nothing is close (distance >= 5)."""
    d = {t: tree_dist(lang, t) for t in MAYAN}
    m = min(d.values())
    return None if m >= 5 else sorted(t for t, v in d.items() if v == m)


def branch_proba(P):
    return np.stack([P[:, [LANGS.index(l) for l in LANGS if BRANCH[l] == b]].sum(1)
                     for b in BRANCHES], 1)


# ------------------------------------------------------------------ shared setup
def setup():
    man = pd.read_parquet(MANIFEST)
    clean, F = load_view("clean", man)
    aug, FA = load_view("aug", man)
    btr = np.where((clean.role == "train") & (clean.corpus == "bloom"))[0]
    yb, gb = clean.label.values[btr], clean.group.values[btr]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(btr, yb, gb))
    pools, test_groups = split_new_corpora(clean)
    return man, clean, F, aug, FA, btr, yb, gb, folds, pools, test_groups


def oof_embedding(clean, F, aug, FA, btr, folds):
    P = np.zeros((len(btr), len(LANGS)))
    for tr, te in folds:
        X, y = training_set(FINAL, clean, F, aug, FA, btr[tr])
        P[te] = proba(make_clf().fit(X, y), F["lid"][btr[te]])
    return P


# ------------------------------------------------------------------ A1 several utterances
def aggregate(P, groups, n, seed=SEED):
    """Disjoint sets of n clips from the same group; geometric-mean of probabilities per set."""
    rs = np.random.default_rng(seed)
    sets = []
    for g in pd.unique(groups):
        idx = rs.permutation(np.where(groups == g)[0])
        sets += [idx[i:i + n] for i in range(0, len(idx) - n + 1, n)]
    logp = np.stack([np.log(P[s] + 1e-9).mean(0) for s in sets])
    Q = np.exp(logp - logp.max(1, keepdims=True)); Q /= Q.sum(1, keepdims=True)
    return Q, sets


def A1(sources, tau, tau_m, ns=(1, 2, 4, 8)):
    out = {}
    for name, (P, y, groups) in sources.items():
        out[name] = {}
        for n in ns:
            Q, sets = aggregate(P, groups, n)
            yy = np.array([y[s[0]] for s in sets])
            pred = L[Q.argmax(1)]
            r = routing_metrics(yy, Q, tau, tau_m)
            out[name][n] = {"sets": len(sets), "accuracy": float((pred == yy).mean()),
                            "useful_answer": r.get("useful_answer"),
                            "mayan_routed_to_spanish": r.get("mayan_routed_to_spanish")}
        print("A1", name, {n: round(v["accuracy"], 3) for n, v in out[name].items()})
    return out


# ------------------------------------------------------------------ A2 concept erasure
def leace(X, Z, within=None):
    """LEACE eraser for concept Z (one-hot). If `within` (labels) is given, X and Z are centred
    inside each label first, so only corpus differences *within the same language* are erased.
    Returns (mu, E); erase with x - (x - mu) @ E.T. Rank of E <= number of concept classes - 1."""
    mu = X.mean(0)
    Xc, Zc = X - mu, Z - Z.mean(0)
    if within is not None:
        Xc, Zc = Xc.copy(), Zc.copy()
        for w in np.unique(within):
            m = within == w
            Xc[m] -= Xc[m].mean(0); Zc[m] -= Zc[m].mean(0)
    n = len(X)
    sxx, sxz = Xc.T @ Xc / n, Xc.T @ Zc / n
    ev, V = eigh(sxx)
    ev = np.clip(ev, ev.max() * 1e-6, None)
    W, Winv = (V / np.sqrt(ev)) @ V.T, (V * np.sqrt(ev)) @ V.T
    U, S, _ = np.linalg.svd(W @ sxz, full_matrices=False)
    U = U[:, S > S.max() * 1e-6]
    return mu, Winv @ (U @ U.T) @ W, U.shape[1]


def random_eraser(X, rank, seed=SEED):
    """Control: orthogonal projection removing `rank` random directions."""
    Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(X.shape[1], rank)))
    return X.mean(0), Q @ Q.T


def apply(X, mu, E):
    return X - (X - mu) @ E.T


def corpus_probe_auroc(X, corpus, groups):
    """How well a linear probe tells Bloom from the other corpora (grouped CV)."""
    yc = (corpus != "bloom").astype(int)
    if yc.min() == yc.max():
        return None
    P = np.zeros(len(yc))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, yc, groups):
        c = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        P[te] = c.fit(X[tr], yc[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(yc, P))


def A2(clean, F, btr, folds, pools):
    out = {}
    xm = (clean.role == "xcorpus").values
    bloom_test = btr[folds[0][1]]                       # one fold of Bloom books, never trained on
    bloom_train = btr[folds[0][0]]
    for held in XCORPORA:
        tr_idx = [bloom_train]
        if held != "fleurs":
            tr_idx.append(np.where((clean.role == "train_extra").values)[0])
        for c in XCORPORA:
            if c != held:
                tr_idx.append(np.where(xm & clean.group.isin(pools[c]).values)[0])
        tr = np.concatenate(tr_idx)
        te = np.where(xm & (clean.corpus == held).values)[0]
        X, y, corp = F["lid"][tr], clean.label.values[tr], clean.corpus.values[tr]
        Z = pd.get_dummies(corp).values.astype(float)
        mu_w, E_w, rank = leace(X, Z, within=y)
        mu_n, E_n, _ = leace(X, Z)
        mu_r, E_r = random_eraser(X, rank)
        res = {"train_corpora": sorted(set(corp)), "erased_rank": rank, "n_test": int(len(te))}
        for cond, (mu, E) in {"none": (X.mean(0), np.zeros((X.shape[1],) * 2)),
                              "leace_within_language": (mu_w, E_w),
                              "leace_naive": (mu_n, E_n), "random_same_rank": (mu_r, E_r)}.items():
            clf = make_clf().fit(apply(X, mu, E), y)
            Pt = proba(clf, apply(F["lid"][te], mu, E))
            Pb = proba(clf, apply(F["lid"][bloom_test], mu, E))
            yt = clean.label.values[te]
            pt = L[Pt.argmax(1)]
            yb_ = clean.label.values[bloom_test]
            res[cond] = {
                "heldout_corpus_accuracy": float((pt == yt).mean()),
                "heldout_spanish_vs_mayan": float((np.isin(pt, MAYAN) == np.isin(yt, MAYAN)).mean()),
                "bloom_heldout_books_macro_f1": float(f1_score(yb_, L[Pb.argmax(1)], average="macro")),
            }
            if cond in ("none", "leace_within_language"):
                sp = np.where(np.isin(y, ["spa", "mam", "quc", "cak"]))[0][::3]
                res[cond]["corpus_probe_auroc_train"] = corpus_probe_auroc(
                    apply(X[sp], mu, E), corp[sp], clean.group.values[tr][sp])
        out[held] = res
        print("A2", held, {k: round(v["heldout_corpus_accuracy"], 3)
                           for k, v in res.items() if isinstance(v, dict)})
    return out


def nap(X, lang, group, k):
    """Nuisance attribute projection: remove the top-k directions of variation *between recordings
    of the same language* (speaker, microphone, room, narrator). Classic in speaker/language ID."""
    mu = X.mean(0)
    rows = []
    for l in np.unique(lang):
        ml = lang == l
        ctr = X[ml].mean(0)
        for g in np.unique(group[ml]):
            mg = ml & (group == g)
            rows.append(np.sqrt(mg.sum()) * (X[mg].mean(0) - ctr))
    _, _, Vt = np.linalg.svd(np.stack(rows), full_matrices=False)
    U = Vt[:k].T
    return mu, U @ U.T


def A2b(clean, F, btr, folds, pools, ks=(0, 10, 25, 50, 100, 200)):
    """Leave-one-corpus-out with NAP at several k (reported as a sweep: k is not tuned on the
    left-out corpus), plus a random-direction control at k=50."""
    out = {}
    xm = (clean.role == "xcorpus").values
    bloom_test, bloom_train = btr[folds[0][1]], btr[folds[0][0]]
    for held in XCORPORA:
        tr_idx = [bloom_train]
        if held != "fleurs":
            tr_idx.append(np.where((clean.role == "train_extra").values)[0])
        for c in XCORPORA:
            if c != held:
                tr_idx.append(np.where(xm & clean.group.isin(pools[c]).values)[0])
        tr = np.concatenate(tr_idx)
        te = np.where(xm & (clean.corpus == held).values)[0]
        X, y, grp = F["lid"][tr], clean.label.values[tr], clean.group.values[tr]
        res = {}
        for k in list(ks) + ["random50"]:
            if k == 0:
                mu, E = X.mean(0), np.zeros((X.shape[1],) * 2)
            elif k == "random50":
                mu, E = random_eraser(X, 50)
            else:
                mu, E = nap(X, y, grp, k)
            clf = make_clf().fit(apply(X, mu, E), y)
            pt = L[proba(clf, apply(F["lid"][te], mu, E)).argmax(1)]
            pb = L[proba(clf, apply(F["lid"][bloom_test], mu, E)).argmax(1)]
            yt = clean.label.values[te]
            res[str(k)] = {"heldout_corpus_accuracy": float((pt == yt).mean()),
                           "bloom_heldout_books_macro_f1": float(f1_score(clean.label.values[bloom_test], pb, average="macro"))}
        out[held] = res
        print("A2b", held, {k: round(v["heldout_corpus_accuracy"], 3) for k, v in res.items()})
    return out


# ------------------------------------------------------------------ A3 family tree
def mean_error_distance(y, pred):
    err = y != pred
    if not err.any():
        return None, None
    d = np.mean([tree_dist(a, b) for a, b in zip(y[err], pred[err])])
    # expectation if each error picked a wrong class uniformly at random
    rnd = np.mean([np.mean([tree_dist(a, b) for b in LANGS if b != a]) for a in y[err]])
    return float(d), float(rnd)


def fit_hier(X, y):
    """Branch classifier + K'iche'-vs-Kaqchikel classifier (the only multi-language branch)."""
    yb = np.array([BRANCH[l] for l in y])
    bclf = make_clf().fit(X, yb)
    k = np.isin(y, ["quc", "cak"])
    kclf = make_clf().fit(X[k], y[k])
    return bclf, kclf


def hier_proba(models, X):
    bclf, kclf = models
    Pb = bclf.predict_proba(X)[:, [list(bclf.classes_).index(b) for b in BRANCHES]]
    Pk = kclf.predict_proba(X)[:, [list(kclf.classes_).index(l) for l in ["quc", "cak"]]]
    P = np.zeros((len(X), len(LANGS)))
    for l in LANGS:
        b = BRANCHES.index(BRANCH[l])
        P[:, LANGS.index(l)] = Pb[:, b] * (Pk[:, ["quc", "cak"].index(l)] if l in ("quc", "cak") else 1)
    return P


def pick_tau_branch(y, P, target=0.95):
    Pb = branch_proba(P)
    conf, pred = Pb.max(1), np.array(BRANCHES)[Pb.argmax(1)]
    yt = np.array([BRANCH[l] for l in y])
    for t in np.round(np.arange(0.5, 0.995, 0.01), 2):
        a = conf >= t
        if a.any() and (pred[a] == yt[a]).mean() >= target:
            return float(t)
    return 0.99


def decide(P, tau, tau_b, tau_m):
    """language -> branch -> 'Mayan' -> not sure."""
    Pb = branch_proba(P)
    mi = [LANGS.index(l) for l in MAYAN]
    out = []
    for p, pb in zip(P, Pb):
        if p.max() >= tau:
            out.append(("lang", LANGS[int(p.argmax())]))
        elif pb.max() >= tau_b and BRANCHES[int(pb.argmax())] not in ("Spanish", "English"):
            out.append(("branch", BRANCHES[int(pb.argmax())]))
        elif p[mi].sum() >= tau_m:
            out.append(("mayan", None))
        else:
            out.append(("unsure", None))
    return out


def score_decisions(y, dec):
    """Share of target-language clips that got a useful answer (named correctly, or a branch that
    contains the language), and of those routed to a wrong place."""
    useful = [(k == "lang" and v == t) or (k == "branch" and BRANCH[t] == v)
              for t, (k, v) in zip(y, dec)]
    kinds = pd.Series([k for k, _ in dec]).value_counts(normalize=True).to_dict()
    return {"useful_answer": float(np.mean(useful)), **{f"share_{k}": float(v) for k, v in kinds.items()}}


def score_openset(lang, dec):
    near = nearest_targets(lang)
    cats = []
    for k, v in dec:
        if k == "unsure":
            cats.append("declined")
        elif k == "mayan":
            cats.append("said Mayan")
        elif k == "branch":
            cats.append("nearest branch" if near and BRANCH[near[0]] == v else "other branch")
        elif v in ("spa", "eng"):
            cats.append("said Spanish/English")
        else:
            cats.append("named nearest relative" if near and v in near else "named unrelated language")
    return {"nearest_covered_relatives": near,
            **pd.Series(cats).value_counts(normalize=True).round(3).to_dict()}


def conformal(P_cal, y_cal, alpha=0.1):
    s = 1 - P_cal[np.arange(len(y_cal)), [LANGS.index(t) for t in y_cal]]
    n = len(s)
    return float(np.quantile(s, min(1, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def conformal_eval(P, y, q):
    sets = P >= 1 - q
    cover = sets[np.arange(len(y)), [LANGS.index(t) for t in y]]
    return {"coverage": float(cover.mean()), "mean_set_size": float(sets.sum(1).mean())}


def A3(clean, F, aug, FA, btr, yb, gb, folds, oofP, pools, test_groups, tau, tau_m):
    out = {}
    xm = (clean.role == "xcorpus").values
    # flat vs hierarchical, Bloom held-out books
    Ph = np.zeros_like(oofP)
    for tr, te in folds:
        X, y = training_set(FINAL, clean, F, aug, FA, btr[tr])
        Ph[te] = hier_proba(fit_hier(X, y), F["lid"][btr[te]])
    X, y = training_set(FINAL, clean, F, aug, FA, btr)
    flat, hier = make_clf().fit(X, y), fit_hier(X, y)
    xc = np.where(xm & (clean.corpus != "fleurs").values)[0]
    yx = clean.label.values[xc]
    Pxf, Pxh = proba(flat, F["lid"][xc]), hier_proba(hier, F["lid"][xc])
    for name, P, yy in [("bloom_heldout_books_flat", oofP, yb), ("bloom_heldout_books_hier", Ph, yb),
                        ("new_corpora_flat", Pxf, yx), ("new_corpora_hier", Pxh, yx)]:
        pred = L[P.argmax(1)]
        bp = np.array(BRANCHES)[branch_proba(P).argmax(1)]
        d, rnd = mean_error_distance(yy, pred)
        out[name] = {"accuracy": float((pred == yy).mean()),
                     "branch_accuracy": float((bp == np.array([BRANCH[t] for t in yy])).mean()),
                     "mean_tree_distance_of_errors": d, "if_errors_were_random": rnd}
    print("A3 tree", {k: (round(v["accuracy"], 3), round(v["branch_accuracy"], 3)) for k, v in out.items()})

    # branch fallback decision (flat probabilities), thresholds from Bloom held-out books
    tau_b = pick_tau_branch(yb, oofP)
    out["branch_threshold"] = tau_b
    out["decision_bloom_heldout"] = score_decisions(yb, decide(oofP, tau, tau_b, tau_m))
    out["decision_new_corpora_bloom_only"] = {
        c: score_decisions(yx[clean.corpus.values[xc] == c],
                           decide(Pxf[clean.corpus.values[xc] == c], tau, tau_b, tau_m))
        for c in ["coerll", "tachiwin", "tzijonel"]}

    # shipped-style model (adapted with half of each new corpus) on open-set languages
    ad = np.where(xm & clean.group.isin(set().union(*map(set, pools.values()))).values)[0]
    shipped = make_clf().fit(np.concatenate([X, F["lid"][ad]]), np.concatenate([y, clean.label.values[ad]]))
    om = (clean.role == "openset").values
    Po = proba(shipped, F["lid"][om])
    labs = clean.label.values[om]
    out["openset_shipped_model"] = {NAMES.get(l, l): score_openset(l, decide(Po[labs == l], tau, tau_b, tau_m))
                                    for l in pd.unique(labs)}

    # split conformal: calibrate on half of the Bloom held-out books, test elsewhere
    ub = np.array(sorted(set(gb)))
    cal_books = set(np.random.default_rng(SEED).permutation(ub)[: len(ub) // 2])
    cal = np.isin(gb, list(cal_books))
    q = conformal(oofP[cal], yb[cal])
    te_ = np.where(xm & clean.group.isin(test_groups).values & (clean.corpus != "fleurs").values)[0]
    out["conformal_90"] = {
        "threshold_q": q,
        "bloom_other_half_books": conformal_eval(oofP[~cal], yb[~cal], q),
        "new_corpora_bloom_only_model": {c: conformal_eval(Pxf[clean.corpus.values[xc] == c],
                                                           yx[clean.corpus.values[xc] == c], q)
                                         for c in ["coerll", "tachiwin", "tzijonel"]},
        "new_corpora_heldout_adapted_model": {c: conformal_eval(
            proba(shipped, F["lid"][te_][clean.corpus.values[te_] == c]),
            clean.label.values[te_][clean.corpus.values[te_] == c], q)
            for c in ["coerll", "tachiwin", "tzijonel"]},
    }
    print("A3 conformal", out["conformal_90"]["bloom_other_half_books"])
    return out, Pxf, xc, shipped


# ------------------------------------------------------------------ A4 phonotactics
def phone_vectorizer():
    return TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 3), sublinear_tf=True, min_df=3,
                           lowercase=False)


def phone_clf():
    return LogisticRegression(C=3.0, max_iter=3000, class_weight="balanced")


def phone_proba(vec, clf, texts):
    p = clf.predict_proba(vec.transform(texts))
    return p[:, [list(clf.classes_).index(l) for l in LANGS]]


def A4(clean, F, aug, FA, btr, yb, folds, oofP, Pxf, xc):
    f = FEATS / "phones.parquet"
    if not f.exists():
        print("A4 skipped: run python -m wi.phones first"); return None
    ph = pd.read_parquet(f).drop_duplicates("clip_id").set_index("clip_id").phones
    T = clean.clip_id.map(ph).fillna("").values
    extra = np.where((clean.role == "train_extra").values)[0]
    # held-out books (same folds as the embedding model)
    Pp = np.zeros_like(oofP)
    for tr, te in folds:
        tri = np.concatenate([btr[tr], extra])
        vec = phone_vectorizer().fit(T[tri])
        clf = phone_clf().fit(vec.transform(T[tri]), clean.label.values[tri])
        Pp[te] = phone_proba(vec, clf, T[btr[te]])
    # fusion weight chosen on held-out Bloom books only
    def fuse(Pa, Pb, w):
        lg = w * np.log(Pa + 1e-9) + (1 - w) * np.log(Pb + 1e-9)
        Q = np.exp(lg - lg.max(1, keepdims=True)); return Q / Q.sum(1, keepdims=True)
    grid = np.round(np.linspace(0, 1, 11), 1)
    f1s = {w: f1_score(yb, L[fuse(oofP, Pp, w).argmax(1)], average="macro") for w in grid}
    w = max(f1s, key=f1s.get)
    tri = np.concatenate([btr, extra])
    vec = phone_vectorizer().fit(T[tri])
    clf = phone_clf().fit(vec.transform(T[tri]), clean.label.values[tri])
    Ppx = phone_proba(vec, clf, T[xc])
    yx = clean.label.values[xc]
    out = {"fusion_weight_on_embedding": float(w),
           "bloom_heldout_books_macro_f1": {"embedding": float(f1s[1.0]), "phones": float(f1s[0.0]),
                                            "fusion": float(f1s[w])}}
    out["new_corpora_accuracy"] = {}
    for c in ["coerll", "tachiwin", "tzijonel"]:
        m = clean.corpus.values[xc] == c
        out["new_corpora_accuracy"][c] = {
            "embedding": float((L[Pxf[m].argmax(1)] == yx[m]).mean()),
            "phones": float((L[Ppx[m].argmax(1)] == yx[m]).mean()),
            "fusion": float((L[fuse(Pxf[m], Ppx[m], w).argmax(1)] == yx[m]).mean())}
    # does the phone channel leak the recording corpus less? (Spanish: Bloom vs FLEURS)
    sm = (clean.label == "spa").values & clean.corpus.isin(["bloom", "fleurs"]).values
    ys = (clean.corpus.values[sm] == "fleurs").astype(int)
    Ps = np.zeros(sm.sum())
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(T[sm], ys, clean.group.values[sm]):
        v = phone_vectorizer().fit(T[sm][tr])
        c = LogisticRegression(C=3.0, max_iter=3000).fit(v.transform(T[sm][tr]), ys[tr])
        Ps[te] = c.predict_proba(v.transform(T[sm][te]))[:, 1]
    out["corpus_probe_auroc_phones_spanish_bloom_vs_fleurs"] = float(roc_auc_score(ys, Ps))
    # interpretability: strongest phone n-grams per language
    names = np.array(vec.get_feature_names_out())
    out["top_phone_ngrams"] = {NAMES[l]: list(names[np.argsort(clf.coef_[list(clf.classes_).index(l)])[::-1][:10]])
                               for l in LANGS}
    out["mayan_marked_phones_share_of_clips"] = {
        NAMES[l]: {p: float(np.mean([p in t.split() for t in T[btr][yb == l]])) for p in ["q", "ʔ", "χ"]}
        for l in LANGS}
    print("A4", out["bloom_heldout_books_macro_f1"], out["new_corpora_accuracy"])
    return out


# ------------------------------------------------------------------ report
def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def figures(R):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, d in R["A1_several_utterances"].items():
        ns = sorted(int(k) for k in d)
        ax.plot(ns, [d[n]["accuracy"] if n in d else d[str(n)]["accuracy"] for n in ns], marker="o", label=name)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_xlabel("clips from the same recording"); ax.set_ylabel("accuracy (7 classes)")
    ax.set_ylim(0, 1); ax.legend(fontsize=7); ax.set_title("More speech from the same person")
    fig.tight_layout(); fig.savefig(RESULTS / "several_utterances.png", dpi=160); plt.close(fig)

    conds = ["none", "random_same_rank", "leace_naive", "leace_within_language"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, c in enumerate(conds):
        ax.bar(np.arange(len(XCORPORA)) + i * 0.2,
               [R["A2_concept_erasure"][h][c]["heldout_corpus_accuracy"] for h in XCORPORA], 0.2,
               label=c.replace("_", " "))
    tick = {"coerll": "COERLL\nK'iche'", "tachiwin": "Tachiwin\nMam", "tzijonel": "Tzijonel\nKaqchikel",
            "fleurs": "FLEURS\nSpa/Eng", "google_es": "Google\nSpanish", "librispeech": "LibriSpeech\nEnglish"}
    ax.set_xticks(np.arange(len(XCORPORA)) + 0.3, [tick.get(c, c) for c in XCORPORA], fontsize=7)
    ax.set_ylabel("accuracy on a corpus left out of training"); ax.set_ylim(0, 1); ax.legend(fontsize=7)
    ax.set_title("Erasing recording-corpus information")
    fig.tight_layout(); fig.savefig(RESULTS / "concept_erasure.png", dpi=160); plt.close(fig)


def report(R):
    out = ["# Advanced methods\n", "## A1 Several utterances from the same speaker\n",
           "Clips from one recording (book, episode, lesson) are grouped into disjoint sets of n; "
           "evidence is combined by averaging log-probabilities.\n",
           "| source | n=1 | n=2 | n=4 | n=8 | useful answer n=1 -> n=8 |", "|---|---|---|---|---|---|"]
    for name, d in R["A1_several_utterances"].items():
        g = lambda n, k: d[n][k] if n in d else d[str(n)][k]
        out.append(f"| {name} | " + " | ".join(pct(g(n, "accuracy")) for n in [1, 2, 4, 8]) +
                   f" | {pct(g(1, 'useful_answer'))} -> {pct(g(8, 'useful_answer'))} |")
    out += ["", "## A2 Erasing the recording corpus (LEACE), leave-one-corpus-out\n",
            "| corpus left out | no erasure | random directions | naive LEACE | within-language LEACE | Bloom books macro-F1 (none -> within) | corpus probe AUROC (none -> within) |",
            "|---|---|---|---|---|---|---|"]
    for h, r in R["A2_concept_erasure"].items():
        a = lambda c: pct(r[c]["heldout_corpus_accuracy"])
        out.append(f"| {h} | {a('none')} | {a('random_same_rank')} | {a('leace_naive')} | "
                   f"{a('leace_within_language')} | {r['none']['bloom_heldout_books_macro_f1']:.3f} -> "
                   f"{r['leace_within_language']['bloom_heldout_books_macro_f1']:.3f} | "
                   f"{r['none'].get('corpus_probe_auroc_train')} -> {r['leace_within_language'].get('corpus_probe_auroc_train')} |")
    nb = R["A2b_nuisance_projection"]
    ks = list(next(iter(nb.values())).keys())
    out += ["", "### A2b Nuisance attribute projection (remove top-k between-recording directions within each language)\n",
            "Sweep over k (not tuned on the left-out corpus). Accuracy on the corpus left out of training:\n",
            "| corpus left out | " + " | ".join(f"k={k}" for k in ks) + " |", "|---" * (len(ks) + 1) + "|"]
    for h, r in nb.items():
        out.append(f"| {h} | " + " | ".join(pct(r[k]["heldout_corpus_accuracy"]) for k in ks) + " |")
    A3 = R["A3_family_tree"]
    out += ["", "## A3 Mayan family tree\n",
            "| setting | accuracy | branch accuracy | mean tree distance of errors | if errors were random |",
            "|---|---|---|---|---|"]
    for k in ["bloom_heldout_books_flat", "bloom_heldout_books_hier", "new_corpora_flat", "new_corpora_hier"]:
        v = A3[k]
        out.append(f"| {k} | {pct(v['accuracy'])} | {pct(v['branch_accuracy'])} | "
                   f"{v['mean_tree_distance_of_errors']:.2f} | {v['if_errors_were_random']:.2f} |")
    out += ["", f"Branch fallback threshold {A3['branch_threshold']}. Useful answers "
            f"(right language, or a branch containing it): held-out Bloom books "
            f"{pct(A3['decision_bloom_heldout']['useful_answer'])}; new corpora (Bloom-only model) " +
            ", ".join(f"{c} {pct(v['useful_answer'])}" for c, v in A3['decision_new_corpora_bloom_only'].items()),
            "", "Languages it has no class for (shipped model):\n",
            "| language | nearest covered relative | outcome shares |", "|---|---|---|"]
    for l, v in A3["openset_shipped_model"].items():
        near = v.pop("nearest_covered_relatives")
        out.append(f"| {l} | {', '.join(NAMES[x] for x in near) if near else 'none close'} | {v} |")
    c = A3["conformal_90"]
    out += ["", f"Split-conformal 90% prediction sets (threshold {c['threshold_q']:.3f}): "
            f"other half of Bloom books coverage {pct(c['bloom_other_half_books']['coverage'])}, "
            f"mean set size {c['bloom_other_half_books']['mean_set_size']:.2f}.",
            "New corpora, Bloom-only model: " + ", ".join(
                f"{k} {pct(v['coverage'])} (size {v['mean_set_size']:.2f})"
                for k, v in c["new_corpora_bloom_only_model"].items()),
            "New corpora, held-out recordings, adapted model: " + ", ".join(
                f"{k} {pct(v['coverage'])} (size {v['mean_set_size']:.2f})"
                for k, v in c["new_corpora_heldout_adapted_model"].items())]
    A4 = R.get("A4_phonotactics")
    if A4:
        out += ["", "## A4 Phonotactics (universal phone recogniser + phone n-grams)\n",
                f"Held-out Bloom books macro-F1: embedding {A4['bloom_heldout_books_macro_f1']['embedding']:.3f}, "
                f"phones {A4['bloom_heldout_books_macro_f1']['phones']:.3f}, fusion "
                f"{A4['bloom_heldout_books_macro_f1']['fusion']:.3f} (weight on embedding "
                f"{A4['fusion_weight_on_embedding']}).\n",
                "| new corpus | embedding | phones | fusion |", "|---|---|---|---|"]
        for k, v in A4["new_corpora_accuracy"].items():
            out.append(f"| {k} | {pct(v['embedding'])} | {pct(v['phones'])} | {pct(v['fusion'])} |")
        out += ["", f"Corpus probe on phone n-grams (Bloom vs FLEURS Spanish): AUROC "
                f"{A4['corpus_probe_auroc_phones_spanish_bloom_vs_fleurs']:.3f}", "",
                "Top phone n-grams per language:\n"]
        out += [f"- {k}: {' · '.join(v)}" for k, v in A4["top_phone_ngrams"].items()]
        out += ["", "Share of clips containing uvular /q/, glottal stop /ʔ/, /χ/: " +
                json.dumps(A4["mayan_marked_phones_share_of_clips"], ensure_ascii=False)]
    (RESULTS / "advanced.md").write_text("\n".join(out) + "\n")


def main():
    man, clean, F, aug, FA, btr, yb, gb, folds, pools, test_groups = setup()
    oofP = oof_embedding(clean, F, aug, FA, btr, folds)
    tau, tau_m = pick_tau(yb, oofP), pick_tau_mayan(yb, oofP)
    R = {"tau": tau, "tau_mayan": tau_m}
    R["A3_family_tree"], Pxf, xc, shipped = A3(clean, F, aug, FA, btr, yb, gb, folds, oofP,
                                               pools, test_groups, tau, tau_m)
    xm = (clean.role == "xcorpus").values
    te = np.where(xm & clean.group.isin(test_groups).values & (clean.corpus != "fleurs").values)[0]
    Pte = proba(shipped, F["lid"][te])
    src = {"Bloom held-out books": (oofP, yb, gb)}
    for c in ["coerll", "tachiwin", "tzijonel"]:
        m = clean.corpus.values[xc] == c
        src[f"{c}, Bloom-only model"] = (Pxf[m], clean.label.values[xc][m], clean.group.values[xc][m])
        m2 = clean.corpus.values[te] == c
        src[f"{c}, adapted model, held-out recordings"] = (Pte[m2], clean.label.values[te][m2],
                                                          clean.group.values[te][m2])
    R["A1_several_utterances"] = A1(src, tau, tau_m)
    R["A2_concept_erasure"] = A2(clean, F, btr, folds, pools)
    R["A2b_nuisance_projection"] = A2b(clean, F, btr, folds, pools)
    R["A4_phonotactics"] = A4(clean, F, aug, FA, btr, yb, folds, oofP, Pxf, xc)
    (RESULTS / "metrics_advanced.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=str))
    figures(R)
    report(R)
    print("advanced done")


if __name__ == "__main__":
    main()
