"""The demo's decision rule, shared by app.py and wi/validate.py so they cannot drift apart.

Per utterance (one or more clips from the same person):
  1. Each clip gets 7-class fused probabilities (embedding + phones; embedding only when the phone
     recogniser heard nothing) from router.joblib, and 8-class probabilities (with OTHER) from the
     open-set model in openset.joblib.
  2. Clips are combined by averaging log-probabilities (evidence accumulates), separately for both.
  3. Open-set gate first: if 1 - P(other) of the combined 8-class distribution is below the
     threshold -> "not supported".
  4. Otherwise: name the language if its probability >= tau; else "Mayan, try A then B" if
     P(any Mayan) >= tau_mayan; else "not sure".
"""
import numpy as np

from .config import LANGS, MAYAN, SR, WINDOW_SECONDS

MI = [LANGS.index(l) for l in MAYAN]


def windows(x):
    """Split one recorded answer into clips: whole if <= 8 s, else up to five 6 s windows."""
    win = int(WINDOW_SECONDS * SR)
    if len(x) <= 8 * SR:
        return [x]
    return [x[s:s + win] for s in range(0, len(x) - win // 2, win)][:5]


def _probs(clf, X, labels):
    p = clf.predict_proba(X)
    return p[:, [list(clf.classes_).index(l) for l in labels]]


def fuse_logp(pe, pp, w):
    return w * np.log(pe + 1e-9) + (1 - w) * np.log(pp + 1e-9)


def clip_logprobs(H, phones, router, osm):
    """Per-clip log-probabilities: (L7 [n,7] from the shipped router, L8 [n,8] from the open-set model)."""
    phones = np.asarray(phones, dtype=object)
    has = phones != ""
    pe = _probs(router["pipeline"], H, LANGS)
    L7 = np.log(pe + 1e-9)
    if has.any():
        pp = _probs(router["phone_clf"], router["phone_vectorizer"].transform(phones[has]), LANGS)
        L7[has] = fuse_logp(pe[has], pp, router["fusion_weight"])
    P8 = osm["router8"].proba(H, phones, osm["labels8"])        # handles empty phones itself
    return L7, np.log(P8 + 1e-9)


def combine(L):
    """Average log-probabilities over clips and renormalise."""
    m = L.mean(0)
    p = np.exp(m - m.max())
    return p / p.sum()


def decide(L7, L8, router, osm):
    """Decision for ONE utterance given its clips' log-probabilities."""
    p7, p8 = combine(L7), combine(L8)
    known_score = float(1 - p8[-1])
    top = int(p7.argmax())
    out = {"p7": p7, "p8": p8, "known_score": known_score, "top": LANGS[top], "conf": float(p7[top]),
           "p_mayan": float(p7[MI].sum()), "n_clips": int(len(L7))}
    if known_score < osm["threshold"]:
        out["kind"], out["lang"] = "not supported", None
    elif p7[top] >= router["tau"]:
        out["kind"], out["lang"] = "named", LANGS[top]
    elif out["p_mayan"] >= router["tau_mayan"]:
        out["kind"] = "mayan_try_two"
        out["lang"] = [MAYAN[i] for i in np.argsort(p7[MI])[::-1][:2]]
    else:
        out["kind"], out["lang"] = "not sure", None
    return out


def useful(d, true):
    """Was the decision useful for a speaker of `true`? For supported languages: named correctly, or
    'Mayan, try two' containing it. For unsupported languages: declined ('not supported'/'not sure')."""
    if true in LANGS:
        return (d["kind"] == "named" and d["lang"] == true) or (d["kind"] == "mayan_try_two" and true in d["lang"])
    return d["kind"] in ("not supported", "not sure")
