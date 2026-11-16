"""Extract embeddings for every clip in the manifest, for several "views" of the audio.

Views:
  clean        every clip as recorded
  aug          one random augmentation (noise / reverb / phone) of each training clip
  test_noise   cross-corpus clips + 10 dB background noise   } robustness tests
  test_reverb  cross-corpus clips + simulated room reverb      }
  test_phone   cross-corpus clips through a narrow-band phone line

Output per view: data/features/<view>.npz with clip_id, hidden (N,1280), zs_logits (N,6), mfcc (N,160).
Work is checkpointed in chunks so an interrupted run resumes where it stopped.

Run: python -m wi.features [view ...]
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .audio import augment
from .config import CACHE, FEATS, MANIFEST, SEED, SR
from .embed import Embedder, mfcc_stats

CHUNK = 256
BATCH_AUDIO_SECONDS = 32   # cap audio per GPU batch: bounds activation memory (~1.5 GB) at any clip length
TEST_CAP = 300   # robustness views: at most this many clips per (corpus, language)


def clips_for(view, man):
    if view == "clean":
        return man
    if view == "aug":
        return man[man.role.isin(["train", "train_extra"])]
    sub = man[man.role == "xcorpus"]
    return (sub.groupby(["corpus", "label"], group_keys=False)
               .apply(lambda g: g.sample(min(len(g), TEST_CAP), random_state=SEED)))


def batches(order, lengths):
    """Group length-sorted clips so each batch holds at most BATCH_AUDIO_SECONDS of padded audio."""
    out, cur = [], []
    for j in order:
        padded = np.ceil(lengths[j] / (2 * SR)) * 2      # batches are padded to 2 s buckets
        if cur and (len(cur) + 1) * padded > BATCH_AUDIO_SECONDS:
            out.append(np.array(cur)); cur = []
        cur.append(j)
    if cur:
        out.append(np.array(cur))
    return out


def run(view, emb):
    """Embed every clip of `view` that is not already in a saved chunk (id-based, so clips appended
    to the manifest later are picked up without redoing anything)."""
    man = pd.read_parquet(MANIFEST)
    todo = clips_for(view, man).reset_index(drop=True)
    out_dir = FEATS / view
    out_dir.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    existing = sorted(out_dir.glob("chunk_*.npz"))
    for p in existing:
        done_ids |= set(np.load(p, allow_pickle=True)["clip_id"])
    todo = todo[~todo.clip_id.isin(done_ids)]
    kind = {"aug": "random", "test_noise": "noise", "test_reverb": "reverb",
            "test_phone": "phone"}.get(view)
    t0 = time.time()
    first = len(existing)
    nchunks = (len(todo) + CHUNK - 1) // CHUNK
    for c in range(nchunks):
        f = out_dir / f"chunk_{first + c:04d}.npz"
        part = todo.iloc[c * CHUNK:(c + 1) * CHUNK]

        def load_one(args):
            i, cid = args
            x = np.load(CACHE / f"{cid}.npy").astype(np.float32)
            if kind:   # seeded per clip, so results do not depend on thread scheduling
                x = augment(x, kind, np.random.default_rng(SEED * 100003 + i)).astype(np.float32)
            return x
        with ThreadPoolExecutor(4) as pool:
            waves = list(pool.map(load_one, zip(part.index, part.clip_id)))
        order = np.argsort([len(w) for w in waves])
        H = np.zeros((len(waves), 1280), np.float32)
        Z = np.zeros((len(waves), len(emb.zs_index)), np.float32)
        for idx in batches(order, [len(w) for w in waves]):
            h, lg = emb([waves[j] for j in idx])
            H[idx] = h
            if lg is not None:
                Z[idx] = lg[:, emb.zs_index]
        M = np.stack([mfcc_stats(w) for w in waves]).astype(np.float32)
        np.savez(f, clip_id=part.clip_id.values, hidden=H, zs_logits=Z, mfcc=M)
        emb.free()
        done = c + 1
        eta = (time.time() - t0) / done * (nchunks - done) / 60
        print(f"[{view}] chunk {done}/{nchunks}  eta {eta:.0f} min", flush=True)
    merge(view)


def merge(view):
    parts = sorted((FEATS / view).glob("chunk_*.npz"))
    data = [np.load(p, allow_pickle=True) for p in parts]
    np.savez(FEATS / f"{view}.npz",
             **{k: np.concatenate([d[k] for d in data]) for k in ["clip_id", "hidden", "zs_logits", "mfcc"]})
    print(f"[{view}] merged {sum(len(d['clip_id']) for d in data)} clips")


def load(view):
    d = np.load(FEATS / f"{view}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    views = sys.argv[1:] or ["clean", "aug", "test_noise", "test_reverb", "test_phone"]
    emb = Embedder()
    for v in views:
        if v != "clean" and emb.full:
            emb.truncate()          # zero-shot logits are only needed for the clean view
        run(v, emb)
