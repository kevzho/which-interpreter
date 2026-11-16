"""Near-duplicate audio across recordings (re-encoded jingles, intros, re-used recordings).

wi.hygiene removes byte-identical copies; this catches copies that were re-encoded or shifted, which
the Nov 15 audit found leaking across the 50/50 new-corpus split (e.g. Tachiwin jingles, Tzijonel
intros). For every clip: top-5 cosine neighbours on centred MMS-LID layer-24 embeddings among clips of
OTHER groups, then waveform normalised cross-correlation (4 kHz, any lag). NCC >= 0.8 means the same
audio (unrelated speech has median NCC ~0.08). Every clip in such a pair is excluded, as for byte copies.

Run after features (needs data/features/clean.npz): python -m wi.neardup
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from .config import CACHE, FEATS, MANIFEST

NCC_SAME = 0.8
K = 5


def wav(cid):
    x = np.load(CACHE / f"{cid}.npy").astype(np.float32)[::4]
    return (x - x.mean()) / (x.std() + 1e-9)


def ncc(a, b):
    return float(np.abs(fftconvolve(a, b[::-1], mode="full")).max() / min(len(a), len(b)))


def main():
    man = pd.read_parquet(MANIFEST)
    d = np.load(FEATS / "clean.npz", allow_pickle=True)
    ids = pd.Series(d["clip_id"])
    keep = ~ids.duplicated().values
    rows = np.where(keep)[0]
    df = (pd.DataFrame({"clip_id": ids.values[rows], "row": rows})
            .merge(man[["clip_id", "group", "label", "role"]], on="clip_id"))   # drops clips no longer in the manifest
    df = df[df.role.ne("excluded")].reset_index(drop=True)
    H = d["hidden"][df.row.values].astype(np.float32)
    H -= H.mean(0)
    H /= np.linalg.norm(H, axis=1, keepdims=True)
    grp = df.group.values
    pairs = []
    for s in range(0, len(df), 1024):
        S = H[s:s + 1024] @ H.T
        rows = np.arange(s, min(s + 1024, len(df)))
        S[np.arange(len(rows)), rows] = -1                       # not itself
        same_group = grp[rows][:, None] == grp[None, :]
        S[same_group] = -1                                        # only other recordings
        top = np.argpartition(-S, K, axis=1)[:, :K]
        for r, q in enumerate(rows):
            for t in top[r]:
                if q < t:
                    pairs.append((q, t))
    pairs = sorted(set(pairs))
    print(f"{len(pairs)} candidate pairs", flush=True)
    with ThreadPoolExecutor(4) as pool:
        scores = list(pool.map(lambda p: ncc(wav(df.clip_id[p[0]]), wav(df.clip_id[p[1]])), pairs, chunksize=256))
    hits = [(p, sc) for p, sc in zip(pairs, scores) if sc >= NCC_SAME]
    bad = sorted({df.clip_id[i] for (q, t), _ in hits for i in (q, t)})
    print(f"{len(hits)} same-audio pairs across recordings -> {len(bad)} clips excluded")
    m = man.clip_id.isin(bad) & man.role.ne("excluded")
    print(man[m].groupby(["role", "corpus", "label"]).size().to_string())
    man.loc[m, "role_orig"] = man.loc[m, "role"]
    man.loc[m, "exclude_reason"] = "near-duplicate audio (NCC>=0.8) in another recording"
    man.loc[m, "role"] = "excluded"
    man.to_parquet(MANIFEST)


if __name__ == "__main__":
    main()
