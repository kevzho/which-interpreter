"""Frozen external validation set: recording sources that played no part in any modelling decision.

  VoxPopuli test (European Parliament, CC0): es, en (known) + fr, it, de, pl, ro, hr, fi, sk, et, lt (unknown)
  CIEMPIESS test (Mexican Spanish radio, CC-BY-SA): Spanish (known)
  EdAcc test (conversational English, many first-language accents, CC-BY-SA): English (known)
  Lingua Libre (Wikimedia Commons, CC-BY-SA): Kaqchikel word recordings by 3 speakers (known, Mayan)
  Wikitongues windows already in the manifest (role "demo", never used): K'iche' (known); Sakapulteko,
      Yucatec, Lacandon, Garifuna (unknown) - re-tagged as role "external"

Rows get role "external" (no training/tuning code reads it) and ext_known True/False.
Run: python -m wi.add_external
"""
import io
import json
import tarfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .add_other import cache
from .config import EXT, MANIFEST, SEED

OUT = EXT / "external"
# Wikimedia rejects Python's default user agent (HTTP 403); identify the client politely
_opener = urllib.request.build_opener()
_opener.addheaders = [("User-Agent", "WhichInterpreter-hackathon/0.1 (student research project; evaluation only)")]
urllib.request.install_opener(_opener)
VP = {"es": "spa", "en": "eng", "fr": "fra", "it": "ita", "de": "deu", "pl": "pol", "ro": "ron",
      "hr": "hrv", "fi": "fin", "sk": "slk", "et": "est", "lt": "lit"}
HF = "https://huggingface.co/datasets"


def get(url, dest):
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(url, dest.with_suffix(dest.suffix + ".part"))
            dest.with_suffix(dest.suffix + ".part").rename(dest)
            return dest
        except Exception as e:
            print(f"retry {attempt + 1}/5 {url}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


def by_speaker(df, spk, n, per=5):
    df = df.groupby(spk, group_keys=False).apply(lambda g: g.head(per))
    return df.sample(min(n, len(df)), random_state=SEED)


def voxpopuli(rows, blobs, n=150):
    with ThreadPoolExecutor(4) as pool:
        files = dict(zip(VP, pool.map(lambda c: get(f"{HF}/facebook/voxpopuli/resolve/main/{c}/test-00000-of-00001.parquet",
                                                    OUT / "voxpopuli" / f"{c}.parquet"), VP)))
    for code, iso in VP.items():
        t = pq.read_table(files[code], columns=["audio_id", "speaker_id", "audio"]).to_pandas()
        for r in by_speaker(t, "speaker_id", n).itertuples():
            rows.append(dict(clip_id=f"ext_vp_{code}_{r.audio_id}", corpus="voxpopuli", label=iso,
                             group=f"vp:{code}:{r.speaker_id}", license="cc0", ext_known=iso in ("spa", "eng")))
            blobs.append(r.audio["bytes"])


def ciempiess(rows, blobs, n=300):
    tsv = get(f"{HF}/ciempiess/ciempiess_test/resolve/main/corpus/files/metadata_test.tsv", OUT / "ciempiess" / "metadata.tsv")
    tgz = get(f"{HF}/ciempiess/ciempiess_test/resolve/main/corpus/speech/test.tar.gz", OUT / "ciempiess" / "test.tar.gz")
    meta = pd.read_csv(tsv, sep="\t")
    idcol = meta.columns[0]
    spkcol = next((c for c in meta.columns if "speaker" in c.lower() and "id" in c.lower()), meta.columns[1])
    pick = by_speaker(meta, spkcol, n, per=15)        # CIEMPIESS test has only 20 speakers
    want = {str(x): str(s) for x, s in zip(pick[idcol], pick[spkcol])}
    with tarfile.open(tgz) as tar:
        for m in tar:
            stem = m.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if m.isfile() and stem in want:
                rows.append(dict(clip_id=f"ext_ciempiess_{stem}", corpus="ciempiess", label="spa",
                                 group=f"ciempiess:{want[stem]}", license="cc-by-sa-4.0", ext_known=True))
                blobs.append(tar.extractfile(m).read())


def edacc(rows, blobs, n=300):
    tree = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/edinburghcstr/edacc/tree/main/data"))
    shards = sorted(x["path"] for x in tree if x["path"].split("/")[-1].startswith("test-"))[:2]
    for s in shards:
        f = get(f"{HF}/edinburghcstr/edacc/resolve/main/{s}", OUT / "edacc" / s.split("/")[-1])
        t = pq.read_table(f).to_pandas()
        t = t[t.audio.map(lambda a: a is not None)]
        # real answers, not backchannels/laughs: >= 4 words once tags like <LAUGH> are removed
        words = t.text.fillna("").str.replace(r"<[^>]+>", "", regex=True).str.split().str.len()
        t = t[words >= 4]
        spk = "speaker" if "speaker" in t.columns else t.columns[0]
        for r in by_speaker(t, spk, n // len(shards), per=8).itertuples():
            rows.append(dict(clip_id=f"ext_edacc_{s[-12:-8]}_{r.Index}", corpus="edacc", label="eng",
                             group=f"edacc:{getattr(r, spk)}", license="cc-by-sa-4.0", ext_known=True))
            blobs.append(r.audio["bytes"])


def lingua_libre_kaqchikel(rows, blobs):
    """Polite Wikimedia client: one batched metadata request per 50 titles, sequential downloads with
    a pause, long back-off on HTTP 429 (Wikimedia rate-limits bursts)."""
    import urllib.error
    api = "https://commons.wikimedia.org/w/api.php"
    q = {"action": "query", "list": "search", "srsearch": 'intitle:"LL-Q35115 (cak)"', "srnamespace": 6,
         "srlimit": 500, "format": "json"}
    titles = [h["title"] for h in json.load(urllib.request.urlopen(f"{api}?{urllib.parse.urlencode(q)}"))["query"]["search"]]
    urls = {}
    for i in range(0, len(titles), 50):
        info = json.load(urllib.request.urlopen(f"{api}?" + urllib.parse.urlencode(
            {"action": "query", "titles": "|".join(titles[i:i + 50]), "prop": "imageinfo", "iiprop": "url", "format": "json"})))
        for page in info["query"]["pages"].values():
            if "imageinfo" in page:
                urls[page["title"]] = page["imageinfo"][0]["url"].split("?")[0]
    blocked = False
    for t, url in sorted(urls.items()):
        dest = OUT / "lingualibre" / urllib.parse.unquote(url.rsplit("/", 1)[-1])
        if not dest.exists() and not blocked:
            dest.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(6):
                try:
                    urllib.request.urlretrieve(url, dest)
                    break
                except urllib.error.HTTPError as e:
                    wait = int(e.headers.get("Retry-After", 0) or 0) or 30 * (attempt + 1)
                    if wait > 120:    # Wikimedia asked us to back off for long: stop and keep what we have
                        print(f"  {e.code}: server asked to wait {wait}s; stopping Lingua Libre downloads", flush=True)
                        blocked = True
                        break
                    print(f"  {e.code} on {dest.name[:40]}, waiting {wait}s", flush=True)
                    time.sleep(wait)
            time.sleep(1.5)
        if not dest.exists():
            continue
        speaker = t.split("-")[2] if t.count("-") >= 2 else "unknown"
        rows.append(dict(clip_id=f"ext_ll_cak_{dest.stem[:60]}", corpus="lingualibre", label="cak",
                         group=f"ll:{speaker}", license="cc-by-sa-4.0", ext_known=True))
        blobs.append(dest.read_bytes())


def main():
    man = pd.read_parquet(MANIFEST)
    # idempotent: rebuild only the downloaded sources; Wikitongues rows are kept (re-tagged below)
    man = man[~((man.role == "external") & ~man.corpus.eq("wikitongues"))]
    rows, blobs = [], []
    for step in [voxpopuli, ciempiess, edacc, lingua_libre_kaqchikel]:
        n0 = len(rows); step(rows, blobs); print(f"{step.__name__}: {len(rows) - n0} clips", flush=True)
    with ThreadPoolExecutor(16) as pool:
        ok = list(pool.map(lambda rb: cache(rb[0]["clip_id"], rb[1]), zip(rows, blobs)))
    new = pd.DataFrame([{**r, "role": "external", "src": "", "start": None, "dur": None, "note": "external validation",
                         "os_split": ""} for r, good in zip(rows, ok) if good])
    # Wikitongues windows were never used for anything: re-tag them as external
    wt = man.corpus.eq("wikitongues")
    man.loc[wt, "role"] = "external"
    man.loc[wt, "ext_known"] = man.loc[wt, "label"].isin(["spa", "quc"])
    man = pd.concat([man, new], ignore_index=True)
    man.to_parquet(MANIFEST)
    ext = man[man.role == "external"]
    print(ext.groupby(["ext_known", "corpus", "label"]).agg(clips=("clip_id", "size"), groups=("group", "nunique")).to_string())


if __name__ == "__main__":
    main()
