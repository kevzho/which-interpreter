"""Build one manifest over every corpus and cache each clip as 16 kHz float16 audio.

Roles:
  train    - Bloom-Speech target languages (the only data the router is fit on, plus FLEURS-train
             Spanish in the multi-corpus variant)
  xcorpus  - same target languages from a *different* corpus (never trained on): cross-corpus test
  openset  - languages the router must abstain on
  demo     - long-form clips for the live demo only (no labels used in metrics)

Run: python -m wi.prepare
"""
import csv
import io
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

from .audio import decode, duration, speech_windows
from .config import (BLOOM, BLOOM_ENG, CACHE, EXT, MANIFEST, MAX_SECONDS, NOISE, SEED, SR,
                     WINDOW_SECONDS)

rng = np.random.default_rng(SEED)
rows = []


def add(clip_id, corpus, label, role, group, license, src, start=None, dur=None, blob=None, note=""):
    rows.append(dict(clip_id=clip_id, corpus=corpus, label=label, role=role, group=group,
                     license=license, src=str(src) if src is not None else "", start=start,
                     dur=dur, note=note, _blob=blob))


def cap_per_group(items, key, cap, total=None):
    """Keep at most `cap` items per group (spread evenly), then optionally `total` overall."""
    by = defaultdict(list)
    for it in items:
        by[key(it)].append(it)
    out = []
    for g in by.values():
        if len(g) > cap:
            g = [g[i] for i in np.linspace(0, len(g) - 1, cap).round().astype(int)]
        out += g
    if total and len(out) > total:
        out = [out[i] for i in sorted(rng.choice(len(out), total, replace=False))]
    return out


# ------------------------------------------------------------------ Bloom-Speech (SIL Bloom Library)
def bloom():
    for r in csv.DictReader(open(BLOOM / "manifest_clean.csv")):
        if r["exclude"] == "True":
            continue
        role = "openset" if r["label"] == "tzj" else "train"
        add(f"bloom_{r['label']}_{r['file'].split('/')[-1][:-4]}", "bloom", r["label"], role,
            f"bloom:{r['book']}", r["license"], BLOOM / r["file"],
            note="relabeled" if r["relabeled"] == "True" else "")


def bloom_eng(n=400):
    t = pq.read_table(BLOOM_ENG / "test" / "0000.parquet").to_pandas()
    t = t[t.license != "cc-by-nc-nd"]
    items = cap_per_group(list(t.itertuples()), key=lambda r: r.book, cap=3, total=n)
    for r in items:
        add(f"bloomeng_{r.Index}", "bloom", "eng", "openset", f"bloom:{r.book}", r.license, None,
            blob=r.audio["bytes"])


# ------------------------------------------------------------------ FLEURS es_419 (CC-BY-4.0)
def fleurs():
    for split, role, cap in [("test", "xcorpus", None), ("validation", "xcorpus", None),
                             ("train", "train_extra", 1500)]:
        t = pq.read_table(EXT / "fleurs_es_419" / f"{split}.parquet").to_pandas()
        if cap:
            t = t.sample(cap, random_state=SEED)
        for r in t.itertuples():
            add(f"fleurs_{split}_{r.Index}", "fleurs", "spa", role, f"fleurs:{r.id}", "cc-by-4.0",
                None, blob=r.audio["bytes"])


# ------------------------------------------------------------------ Omnilingual ASR corpus (CC-BY-4.0)
def omnilingual():
    for f in sorted((EXT / "omnilingual").glob("*/*.parquet")):
        t = pq.read_table(f).to_pandas()
        for r in t.itertuples():
            add(f"omni_{r.segment_id}", "omnilingual", r.iso_639_3, "openset",
                f"omni:{r.iso_639_3}:{r.speaker_id}", "cc-by-4.0", None, blob=r.audio["bytes"])


# ------------------------------------------------------------------ Tachiwin (INPI Mexico broadcasts)
TACH_OPENSET = ["jac", "yua", "tzh", "tzo", "ctu", "toj", "hus"]


def tachiwin(cap_mam=800, cap_open=150):
    cols = ["code", "acoustic_language", "WOAF", "id"]
    for f in sorted((EXT / "tachiwin" / "data").glob("*.parquet")):
        pf = pq.ParquetFile(f)
        meta = pf.read(columns=cols).to_pandas()
        # Clips the dataset's own acoustic LID tagged as Spanish are Spanish segments inside
        # Indigenous-language broadcasts (see Nov 14 probe notes), so they are not usable labels.
        keep = meta[meta.code.isin(["mam"] + TACH_OPENSET) & (meta.acoustic_language != "es")]
        if keep.empty:
            continue
        audio = pf.read(columns=["audio"]).column("audio").take(list(keep.index)).to_pylist()
        for a, (_, r) in zip(audio, keep.iterrows()):
            role = "xcorpus" if r.code == "mam" else "openset"
            add(f"tach_{r.id}", "tachiwin", r.code, role, f"tach:{r.WOAF}", "none-stated (INPI)",
                None, blob=a["bytes"])
    # cap per language, spreading over recordings
    global rows
    tach = [r for r in rows if r["corpus"] == "tachiwin"]
    rest = [r for r in rows if r["corpus"] != "tachiwin"]
    kept = []
    for code in ["mam"] + TACH_OPENSET:
        items = [r for r in tach if r["label"] == code]
        cap = cap_mam if code == "mam" else cap_open
        kept += cap_per_group(items, key=lambda r: r["group"], cap=max(3, cap // 40), total=cap)
    rows = rest + kept


# ------------------------------------------------------------------ long recordings -> windows
def windows_from(path, max_windows):
    x = decode(path)
    return speech_windows(x, WINDOW_SECONDS, max_windows)


def tzijonel(per_episode=15):
    """Tzijonel Kaqchikel podcast (Internet Archive, CC-BY-NC-ND: evaluation only)."""
    eps = sorted(p for p in (EXT / "tzijonel_kaqchikel").glob("*/*.mp3"))
    with ThreadPoolExecutor(6) as pool:
        wins = list(pool.map(lambda p: windows_from(p, per_episode), eps))
    for p, ws in zip(eps, wins):
        for k, (s, d) in enumerate(ws):
            add(f"tzij_{p.parent.name}_{k}", "tzijonel", "cak", "xcorpus", f"tzij:{p.parent.name}",
                "cc-by-nc-nd-4.0", p, start=s, dur=d, note="podcast window; may contain music/Spanish")


def coerll():
    """UT COERLL K'iche' course audio (CC-BY-4.0): words, phrases, short dialogues."""
    for p in sorted((EXT / "coerll_kiche").glob("*.mp3")):
        d = duration(p)
        if d < 1.0:
            continue
        lesson = "_".join(p.stem.split("_")[:2])
        if d > 2 * WINDOW_SECONDS:
            for k, (s, w) in enumerate(windows_from(p, 6)):
                add(f"coerll_{p.stem}_{k}", "coerll", "quc", "xcorpus", f"coerll:{lesson}",
                    "cc-by-4.0", p, start=s, dur=w)
        else:
            add(f"coerll_{p.stem}", "coerll", "quc", "xcorpus", f"coerll:{lesson}", "cc-by-4.0", p)


def chuj(per_file=60):
    """Chuj corpus (Lett, Zenodo, CC-BY-4.0). Only spontaneous/narrative/story files: the
    elicitation sessions interleave Spanish prompts, and one file is Spanish."""
    for p in sorted((EXT / "chuj_zenodo").glob("*/*")):
        name = p.name
        if p.suffix.lower() not in (".wav", ".mp4") or "Elic" in name or "Spanish" in name:
            continue
        for k, (s, d) in enumerate(windows_from(p, per_file)):
            add(f"chuj_{p.stem.replace(' ', '_')}_{k}", "chuj", "cac", "openset",
                f"chuj:{p.stem}", "cc-by-4.0", p, start=s, dur=d)


def demo():
    """Wikitongues (CC-BY-SA) and Entremundos radio (CC-BY 2.5): demo material only."""
    wt = {"K'iche'": "quc", "Sakapulteko": "quv", "Yucatecan": "yua", "Yucatec": "yua",
          "Lacandon": "lac", "Spanish": "spa", "Garifuna": "cab"}
    import urllib.parse
    for p in sorted((EXT / "wikitongues").glob("*")):
        lab = next((v for k, v in wt.items() if k in urllib.parse.unquote(p.name)), None)
        if lab:
            for k, (s, d) in enumerate(windows_from(p, 8)):
                add(f"wt_{lab}_{p.stem[:30].replace(' ', '_')}_{k}", "wikitongues", lab, "demo",
                    f"wt:{p.stem}", "cc-by-sa", p, start=s, dur=d)
    for p in sorted((EXT / "entremundos").glob("*/*.mp3")):
        for k, (s, d) in enumerate(windows_from(p, 12)):
            add(f"entremundos_{p.parent.name[-12:]}_{k}", "entremundos", "mixed", "demo",
                f"em:{p.parent.name}", "cc-by-2.5", p, start=s, dur=d,
                note="mixed K'iche'/Spanish radio; unlabeled")


# ------------------------------------------------------------------ noise
def unzip_noise():
    NOISE.mkdir(parents=True, exist_ok=True)
    for z in (EXT / "noise").rglob("*.zip"):
        marker = NOISE / (z.stem + ".done")
        if marker.exists():
            continue
        with zipfile.ZipFile(z) as f:
            members = [m for m in f.namelist() if m.lower().endswith(".wav")]
            f.extractall(NOISE / z.stem, members=members)
        marker.touch()


# ------------------------------------------------------------------ cache audio
def cache_clip(r):
    out = CACHE / f"{r['clip_id']}.npy"
    if out.exists():
        return True
    try:
        src = r["_blob"] if r["_blob"] is not None else r["src"]
        x = decode(src, start=r["start"], dur=r["dur"] if r["dur"] else MAX_SECONDS)
        x = x[: int(MAX_SECONDS * SR)]
        if len(x) < int(0.5 * SR):
            return False
        np.save(out, x.astype(np.float16))
        return True
    except Exception as e:
        print("decode failed", r["clip_id"], e)
        return False


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    print("unzipping noise"); unzip_noise()
    for step in [bloom, bloom_eng, fleurs, omnilingual, tachiwin, tzijonel, coerll, chuj, demo]:
        n0 = len(rows); step(); print(f"{step.__name__}: {len(rows) - n0} clips")
    seen = {}
    for r in rows:   # some sources reuse segment ids across splits; make ids unique
        cid = r["clip_id"]
        if cid in seen:
            seen[cid] += 1
            print("duplicate id", cid, r["corpus"])
            r["clip_id"] = f"{cid}__{seen[cid]}"
        else:
            seen[cid] = 0
    print("caching audio ...")
    with ThreadPoolExecutor(8) as pool:
        ok = list(pool.map(cache_clip, rows))
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_blob"} for r in rows])
    df["cached"] = ok
    df = df[df.cached].drop(columns="cached").reset_index(drop=True)
    df.to_parquet(MANIFEST)
    print(df.groupby(["role", "corpus", "label"]).agg(clips=("clip_id", "size"),
                                                      groups=("group", "nunique")).to_string())


if __name__ == "__main__":
    main()
