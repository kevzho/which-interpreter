"""Better "I don't know": outlier exposure + distance-based confidence, with language-disjoint splits.

Protocol (no language is both used and reported):
  fit   : Bloom books of folds 1-4 (+augmentation) + FLEURS-train + adaptation half of each new
          corpus; outlier-exposure methods also see OTHER = languages with os_split == "train"
  tune  : known = Bloom fold-0 books;          unknown = os_split == "tune" languages
          -> choose the method and its threshold (90% of known-language clips answered)
  test  : known = held-out new-corpus recordings + held-out English books;
          unknown = os_split == "test" languages (incl. the six Guatemalan uncovered Mayan languages)

Scores (higher = "this is a language I know"):
  maxprob     max fused probability over the 7 known classes (what the demo used before)
  oe          1 - P(other) from 8-class models trained with OTHER (outlier exposure)
  mahalanobis -min class-conditional Mahalanobis distance (shared Ledoit-Wolf covariance, PCA-256)
  knn         mean cosine similarity to the 10 nearest known-language training clips (PCA-256)
  oe+maha, oe+knn   average of the two scores' ranks (rank fusion, no extra parameters)

Outputs: results/openset.md, results/openset.json, models/openset.joblib (for the demo)
Run: python -m wi.openset   (after wi.add_other, features, phones)
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from .advanced import setup
from .osmodels import Distance, Router
from .config import FEATS, LANGS, MAYAN, MODELS, NAMES, RESULTS, SEED

KNOWN = LANGS
L8 = KNOWN + ["other"]


def rank_avg(a, b):
    """Rank-average two scores over the concatenation of the sets being compared."""
    return (rankdata(a) + rankdata(b)) / 2


def scores_for(r7, r8, dist, Xe, Te):
    P7 = r7.proba(Xe, Te, KNOWN)
    P8 = r8.proba(Xe, Te, L8)
    s = {"maxprob": P7.max(1), "oe": 1 - P8[:, -1], "mahalanobis": dist.mahalanobis(Xe), "knn": dist.knn(Xe)}
    s["oe+maha"] = rank_avg(s["oe"], s["mahalanobis"])
    s["oe+knn"] = rank_avg(s["oe"], s["knn"])
    return s, P7, P8


def evaluate(s_in, s_out):
    out = {}
    for m in s_in:
        if m in ("oe+maha", "oe+knn"):   # rank fusion must be computed jointly over in+out
            a = s_in[m.split("+")[0]]; b = s_in["mahalanobis" if m.endswith("maha") else "knn"]
            c = s_out[m.split("+")[0]]; d = s_out["mahalanobis" if m.endswith("maha") else "knn"]
            joint = rank_avg(np.r_[a, c], np.r_[b, d])
            si, so = joint[: len(a)], joint[len(a):]
        else:
            si, so = s_in[m], s_out[m]
        out[m] = float(roc_auc_score(np.r_[np.ones(len(si)), np.zeros(len(so))], np.r_[si, so]))
    return out


def main():
    man, clean, F, aug, FA, btr, yb, gb, folds, pools, test_groups = setup()
    T = clean.clip_id.map(pd.read_parquet(FEATS / "phones.parquet").drop_duplicates("clip_id").set_index("clip_id").phones).fillna("").values
    w = joblib.load(MODELS / "router.joblib")["fusion_weight"]
    xm = (clean.role == "xcorpus").values
    extra = np.where((clean.role == "train_extra").values)[0]
    # One third of each new corpus's adaptation recordings is held out of the open-set model and used,
    # with held-out Bloom books, to set the "I don't know" threshold: calibration then reflects field
    # recordings, not only in-domain storybooks (the audit found a storybook-only threshold rejected
    # 10-33% of genuine Mam/K'iche'/Kaqchikel recordings).
    rs = np.random.default_rng(SEED + 11)
    cal_groups = set()
    for c, gl in pools.items():
        g = list(rs.permutation(sorted(gl)))
        cal_groups |= set(g[: max(1, len(g) // 3)])
    pool_groups = set().union(*map(set, pools.values()))
    ad = np.where(xm & clean.group.isin(pool_groups - cal_groups).values)[0]
    ad_cal = np.where(xm & clean.group.isin(cal_groups).values)[0]
    osplit = clean.os_split.fillna("").values

    def fit_all(bloom_rows, other_splits):
        known_idx = np.concatenate([bloom_rows, extra, ad])
        Xe_k, y_k = F["lid"][known_idx], clean.label.values[known_idx]
        # augmented copies of the Bloom rows in use and of FLEURS-train (same as the FINAL variant)
        ma = (aug.clip_id.isin(set(clean.clip_id.values[bloom_rows])) | (aug.role == "train_extra")).values
        X_aug, y_aug = FA["lid"][ma], aug.label.values[ma]
        r7 = Router(w).fit(Xe_k, T[known_idx], y_k, X_aug, y_aug)
        oth = np.where(np.isin(osplit, other_splits))[0]
        r8 = Router(w).fit(np.concatenate([Xe_k, F["lid"][oth]]), np.concatenate([T[known_idx], T[oth]]),
                           np.concatenate([y_k, np.array(["other"] * len(oth))]), X_aug, y_aug)
        dist = Distance().fit(Xe_k, y_k)
        return r7, r8, dist, oth

    # ---------------- tune
    r7, r8, dist, oth = fit_all(btr[folds[0][0]], ["train"])
    tin = np.concatenate([btr[folds[0][1]], ad_cal])       # held-out Bloom books + held-out field recordings
    tout = np.where(osplit == "tune")[0]
    s_in, _, _ = scores_for(r7, r8, dist, F["lid"][tin], T[tin])
    s_out, _, _ = scores_for(r7, r8, dist, F["lid"][tout], T[tout])
    tune_auc = evaluate(s_in, s_out)
    best = max(tune_auc, key=tune_auc.get)
    print("tune AUROC", {k: round(v, 3) for k, v in tune_auc.items()}, "-> chosen", best)
    single = best if "+" not in best else None

    # ---------------- test (same fitted models: test languages were never seen)
    tein = np.concatenate([np.where(xm & clean.group.isin(test_groups).values)[0],
                           np.where((clean.role == "heldout_eng").values)[0]])
    teout = np.where(osplit == "test")[0]
    s_tin, P7_in, P8_in = scores_for(r7, r8, dist, F["lid"][tein], T[tein])
    s_tout, P7_out, P8_out = scores_for(r7, r8, dist, F["lid"][teout], T[teout])
    test_auc = evaluate(s_tin, s_tout)
    print("test AUROC", {k: round(v, 3) for k, v in test_auc.items()})
    # per-clip test scores, for the bootstrap in wi.validate
    sc = pd.DataFrame({"known": np.r_[np.ones(len(tein)), np.zeros(len(teout))].astype(bool),
                       "label": np.r_[clean.label.values[tein], clean.label.values[teout]],
                       "group": np.r_[clean.group.values[tein], clean.group.values[teout]],
                       "corpus": np.r_[clean.corpus.values[tein], clean.corpus.values[teout]]})
    for m in ["maxprob", "oe", "mahalanobis", "knn"]:
        sc[m] = np.r_[s_tin[m], s_tout[m]]
    sc["oe+maha"] = rank_avg(sc["oe"].values, sc["mahalanobis"].values)
    # bootstrap clusters: FLEURS "groups" are sentence ids, not speakers -> one cluster per FLEURS language
    sc["cluster"] = np.where(sc.corpus.eq("fleurs") & sc.known, "fleurs:" + sc.label, sc.group)
    sc.to_parquet(RESULTS / "openset_test_scores.parquet")
    # split the unknowns: Guatemalan Mayan (close relatives) vs everything else (distant)
    GT_MAYAN = {"tzj", "poc", "qum", "quv", "jac", "cac"}
    lab_out = clean.label.values[teout]
    close = np.isin(lab_out, list(GT_MAYAN))
    test_auc_close = evaluate(s_tin, {k: v[close] for k, v in s_tout.items()})
    test_auc_far = evaluate(s_tin, {k: v[~close] for k, v in s_tout.items()})
    # every single score at its own tune threshold (answer 90% of tune known clips)
    singles = ["maxprob", "oe", "mahalanobis", "knn"]
    thr_all = {m: float(np.quantile(s_in[m], 0.10)) for m in singles}
    declined_by = {m: {NAMES.get(l, l): float((s_tout[m][lab_out == l] < thr_all[m]).mean())
                    for l in pd.unique(lab_out)} for m in singles}
    answered_known = {m: float((s_tin[m] >= thr_all[m]).mean()) for m in singles}

    # threshold for the chosen single-score method: answer 90% of tune known-language clips
    R = {"tune_auroc": tune_auc, "test_auroc": test_auc, "chosen_on_tune": best,
         "test_auroc_guatemalan_mayan_only": test_auc_close, "test_auroc_distant_only": test_auc_far,
         "declined_by_method": declined_by, "test_known_answered_by_method": answered_known,
         "n": {"tune_known": int(len(tin)), "tune_unknown": int(len(tout)), "test_known": int(len(tein)),
               "test_unknown": int(len(teout)),
               "train_other_languages": int(len(set(clean.label.values[oth]))),
               "tune_languages": sorted(set(clean.label.values[tout])),
               "test_languages": sorted(set(clean.label.values[teout]))}}
    ship_method = single or "oe"          # rank fusion needs a reference set; ship the best single score
    if single is None:
        ship_method = max(["oe", "mahalanobis", "knn", "maxprob"], key=lambda m: tune_auc[m])
    # threshold: answer 90% of the calibration known clips (held-out Bloom books + held-out field recordings)
    thr = float(np.quantile(s_in[ship_method], 0.10))
    yt_in = clean.label.values[tein]
    known_ans = s_tin[ship_method] >= thr
    R["shipped_method"], R["threshold"] = ship_method, thr
    R["test_known_answered"] = float(known_ans.mean())
    R["test_known_answered_by_language"] = {
        NAMES[l]: {"n": int((yt_in == l).sum()), "answered": float(known_ans[yt_in == l].mean()),
                   "answered_correct": float((np.array(KNOWN)[P8_in[:, :-1].argmax(1)][known_ans & (yt_in == l)] == l).mean())
                   if (known_ans & (yt_in == l)).any() else None}
        for l in KNOWN if (yt_in == l).any()}
    R["test_known_accuracy_7class_without_other"] = float((np.array(KNOWN)[P7_in.argmax(1)] == yt_in).mean())
    R["test_known_accuracy_8class_with_other"] = float((np.array(L8)[P8_in.argmax(1)] == yt_in).mean())
    R["test_known_accuracy_when_answered"] = float((np.array(KNOWN)[P8_in[:, :-1].argmax(1)][known_ans] == yt_in[known_ans]).mean())
    per = {}
    for l in pd.unique(lab_out):
        m = lab_out == l
        declined = s_tout[ship_method][m] < thr
        pm = P8_out[m][:, [L8.index(x) for x in MAYAN]].sum(1) / P8_out[m][:, :-1].sum(1)
        per[NAMES.get(l, l)] = {"n": int(m.sum()), "declined": float(declined.mean()),
                                "of_declined_flagged_mayan": float((pm[declined] >= 0.5).mean()) if declined.any() else None,
                                "mayan": l in MAYAN or l in {"tzj", "poc", "qum", "quv", "jac", "cac"}}
    R["test_per_language"] = per

    # ---------------- ship EXACTLY the evaluated open-set model with its own threshold.
    # (An earlier version refit on every OTHER language and reused this threshold; the audit showed the
    # refit model answered only 80% of known-language clips at that threshold, and that the external
    # "unknown" languages had then been trained on. Shipping the evaluated model keeps reported = shipped
    # and keeps tune/test languages unseen.)
    joblib.dump({"router8": r8, "distance": dist, "method": ship_method, "threshold": thr, "labels8": L8,
                 "fit": "Bloom books of folds 1-4 + FLEURS-train + two thirds of the adaptation half of new "
                        "corpora; OTHER = os_split 'train' languages only; threshold set on Bloom fold-0 books + "
                        "the other third of the adaptation half"}, MODELS / "openset.joblib")

    (RESULTS / "openset.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"
    out = ["# Open set: saying \"I don't know\"\n",
           f"Language-disjoint protocol: OTHER trained on {R['n']['train_other_languages']} languages; "
           f"method and threshold chosen on {len(R['n']['tune_languages'])} tune languages; reported on "
           f"{len(R['n']['test_languages'])} test languages never used before "
           f"({', '.join(NAMES.get(l, l) for l in R['n']['test_languages'])}).\n",
           f"| score | tune AUROC | test AUROC (all {len(R['n']['test_languages'])}) | test: 6 Guatemalan Mayan | "
           f"test: {len(R['n']['test_languages']) - 6} other languages |",
           "|---|---|---|---|---|"]
    for m in tune_auc:
        tag = (" (best on tune; not deployable: rank fusion needs a reference set)" if m == best and "+" in m
               else " (shipped)" if m == ship_method else "")
        out.append(f"| {m}{tag} | {tune_auc[m]:.3f} | {test_auc[m]:.3f} | "
                   f"{test_auc_close[m]:.3f} | {test_auc_far[m]:.3f} |")
    out += ["", "Declined at each score's own tune threshold (90% of tune known-language clips answered):\n",
            "| test language | " + " | ".join(singles) + " |", "|---" * (len(singles) + 1) + "|"]
    for l in sorted(declined_by["oe"], key=lambda x: (x not in [NAMES[c] for c in GT_MAYAN], x)):
        out.append(f"| {l} | " + " | ".join(pct(declined_by[m][l]) for m in singles) + " |")
    out.append("| known languages answered (test) | " + " | ".join(pct(answered_known[m]) for m in singles) + " |")
    out += ["", f"Shipped score: **{ship_method}**, threshold set to answer 90% of calibration known-language "
            "clips: held-out Bloom books plus one third of each new corpus's adaptation recordings, held out of "
            "the open-set model (so calibration includes field recordings, not only storybooks). "
            "The demo ships exactly this evaluated model and threshold (no refit), so tune and test "
            "languages remain unseen by the shipped model.",
            f"On test known-language clips it answers {pct(R['test_known_answered'])}, with "
            f"{pct(R['test_known_accuracy_when_answered'])} accuracy on those it answers. Accuracy without "
            f"an OTHER class {pct(R['test_known_accuracy_7class_without_other'])}; with it "
            f"{pct(R['test_known_accuracy_8class_with_other'])}.", "",
            "Known-language test clips answered (not declined), by language:\n",
            "| language | clips | answered | correct when answered |", "|---|---|---|---|"]
    for l, v in R["test_known_answered_by_language"].items():
        out.append(f"| {l} | {v['n']} | {pct(v['answered'])} | {pct(v['answered_correct'])} |")
    out += ["", "| test language | Mayan | clips | declined | of declined, flagged as Mayan |", "|---|---|---|---|---|"]
    for l, v in sorted(per.items(), key=lambda kv: (not kv[1]["mayan"], kv[0])):
        out.append(f"| {l} | {'yes' if v['mayan'] else ''} | {v['n']} | {pct(v['declined'])} | "
                   f"{pct(v['of_declined_flagged_mayan'])} |")
    (RESULTS / "openset.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
