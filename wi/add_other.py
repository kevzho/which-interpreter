"""Add "other language" clips for outlier exposure, and assign every non-target language to exactly
one split so no language is both trained/tuned on and reported:

  os_split = "train"  languages the router may learn as OTHER
  os_split = "tune"   languages used only to choose the open-set method and threshold
  os_split = "test"   languages used only for the final open-set numbers

Mayan: Mexican Mayan (Tachiwin) -> train; Itza' -> tune; the six Guatemalan uncovered languages
(Tz'utujil, Poqomam, Sipakapense, Sakapulteko, Popti', Chuj) -> test.
Non-Mayan: Bloom-Speech's other languages and Tachiwin's non-Mayan language groups, split 60/20/20 by
language (Bloom) and by language group (Tachiwin `superlanguage`, so e.g. all Mixtec varieties land in
one split). Tachiwin clips its own acoustic LID tagged as Spanish are excluded (they are Spanish
segments inside Indigenous-language broadcasts and would teach "Spanish = other").

Run: python -m wi.add_other
"""
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .audio import decode
from .config import BLOOM, CACHE, EXT, MANIFEST, MAX_SECONDS, SEED, SR

TARGET = {"spa", "mam", "quc", "kek", "cak", "kjb", "eng"}
MEX_MAYAN = {"yua", "tzh", "tzo", "ctu", "toj", "hus", "chf"}
TEST_MAYAN = {"tzj", "poc", "qum", "quv", "jac", "cac"}
TUNE_MAYAN = {"itz"}
BLOOM_PER_LANG, BLOOM_PER_BOOK = 60, 6
TACH_PER_CODE, TACH_PER_REC = 30, 3


def split_units(units, seed=SEED):
    u = np.random.default_rng(seed).permutation(sorted(units))
    a, b = int(0.6 * len(u)), int(0.8 * len(u))
    return {x: ("train" if i < a else "tune" if i < b else "test") for i, x in enumerate(u)}


def spread(df, by, per_group, total, seed=SEED):
    df = df.groupby(by, group_keys=False).apply(
        lambda g: g.sample(min(len(g), per_group), random_state=seed))
    return df.sample(min(total, len(df)), random_state=seed)


def cache(cid, src):
    out = CACHE / f"{cid}.npy"
    if out.exists():
        return True
    try:
        x = decode(src)[: int(MAX_SECONDS * SR)]
        if len(x) < int(0.5 * SR):
            return False
        np.save(out, x.astype(np.float16))
        return True
    except Exception:
        return False


def bloom_other():
    data = json.load(open(BLOOM / "bloom_speech.json"))
    info = json.load(open(BLOOM / "book_info.json"))
    langs = sorted(set(data) - TARGET - {"tzj"})
    split = split_units(langs)
    rows = []
    for lang in langs:
        df = pd.DataFrame(data[lang])
        df["license"] = df.book.map(lambda b: info[lang].get(b, {}).get("license"))
        df["quarantine"] = df.book.map(lambda b: info[lang].get(b, {}).get("quarantine"))
        df = df[(df.license != "cc-by-nc-nd") & (df.quarantine != True)].drop_duplicates("url")
        if df.empty:
            continue
        for r in spread(df, "book", BLOOM_PER_BOOK, BLOOM_PER_LANG).itertuples():
            rows.append(dict(clip_id=f"bloomother_{lang}_{os.path.basename(r.url)[:-4]}", corpus="bloom",
                             label=lang, role="other", group=f"bloom:{lang}:{r.book}", license=r.license,
                             src=r.url, start=None, dur=None, note="other language", os_split=split[lang],
                             _src=r.url))
    return rows


def tachiwin_other():
    rows = []
    frames = []
    files = sorted((EXT / "tachiwin" / "data").glob("*.parquet"))
    for f in files:
        m = pq.ParquetFile(f).read(columns=["code", "family", "superlanguage", "acoustic_language",
                                           "WOAF", "id"]).to_pandas()
        m["file"], m["row"] = str(f), np.arange(len(m))
        frames.append(m)
    meta = pd.concat(frames)
    meta = meta[(meta.acoustic_language != "es") & (meta.family != "Mayense")]
    split = split_units(meta.superlanguage.dropna().unique())
    pick = (meta.dropna(subset=["superlanguage"]).groupby("code", group_keys=False)
                .apply(lambda g: spread(g, "WOAF", TACH_PER_REC, TACH_PER_CODE)))
    for f, g in pick.groupby("file"):
        audio = pq.ParquetFile(f).read(columns=["audio"]).column("audio").take(list(g.row)).to_pylist()
        for a, r in zip(audio, g.itertuples()):
            rows.append(dict(clip_id=f"tachother_{r.id}", corpus="tachiwin", label=r.code, role="other",
                             group=f"tach:{r.WOAF}", license="none-stated (INPI)", src="", start=None,
                             dur=None, note=f"other language ({r.superlanguage})",
                             os_split=split[r.superlanguage], _src=a["bytes"]))
    return rows


def tachiwin_chontal():
    """Chontal de Tabasco (Mexican Mayan) was not in the manifest yet."""
    rows = []
    for f in sorted((EXT / "tachiwin" / "data").glob("*.parquet")):
        pf = pq.ParquetFile(f)
        m = pf.read(columns=["code", "acoustic_language", "WOAF", "id"]).to_pandas()
        m["row"] = np.arange(len(m))
        m = m[(m.code == "chf") & (m.acoustic_language != "es")]
        if m.empty:
            continue
        m = spread(m, "WOAF", TACH_PER_REC, 150)
        audio = pf.read(columns=["audio"]).column("audio").take(list(m.row)).to_pylist()
        for a, r in zip(audio, m.itertuples()):
            rows.append(dict(clip_id=f"tach_{r.id}", corpus="tachiwin", label="chf", role="other",
                             group=f"tach:{r.WOAF}", license="none-stated (INPI)", src="", start=None,
                             dur=None, note="Mexican Mayan (Chontal)", os_split="train", _src=a["bytes"]))
    return rows


def main():
    man = pd.read_parquet(MANIFEST)
    man = man[man.role != "other"]                       # idempotent: rebuild the added rows
    # splits for languages already in the manifest
    def existing_split(r):
        if r.label in TARGET or r.role in ("train", "train_extra", "xcorpus", "heldout_eng", "demo",
                                           "external", "excluded"):
            return ""
        if r.label in TEST_MAYAN:
            return "test"
        if r.label in TUNE_MAYAN:
            return "tune"
        if r.label in MEX_MAYAN:
            return "train"
        return ""
    man["os_split"] = [existing_split(r) for r in man.itertuples()]

    new = bloom_other() + tachiwin_other() + tachiwin_chontal()
    with ThreadPoolExecutor(16) as pool:
        ok = list(pool.map(lambda r: cache(r["clip_id"], r["_src"]), new))
    new = pd.DataFrame([{k: v for k, v in r.items() if k != "_src"} for r, good in zip(new, ok) if good])
    new.loc[new.corpus == "bloom", "src"] = ""
    man = pd.concat([man, new], ignore_index=True)
    man.to_parquet(MANIFEST)
    os_rows = man[man.os_split != ""]
    print(os_rows.groupby(["os_split", "corpus"]).agg(clips=("clip_id", "size"),
                                                      languages=("label", "nunique")).to_string())
    print("test languages:", sorted(os_rows[os_rows.os_split == "test"].label.unique()))
    print("tune languages:", sorted(os_rows[os_rows.os_split == "tune"].label.unique()))


if __name__ == "__main__":
    main()
