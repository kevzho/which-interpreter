"""v2 data, added AFTER the frozen v1 external validation showed the open-set gate rejects most Spanish
(and many English) clips from sources it has never heard (88% of VoxPopuli/CIEMPIESS Spanish).

The fix must not use the v1 external set, so v2 adds NEW sources and keeps a NEW blind test:

  diversity for the known classes (role "xcorpus": split by SPEAKER into an adaptation half and a
  held-out half by experiments.split_new_corpora; a third of the adaptation half calibrates the gate)
    google_es    Google crowdsourced Latin-American Spanish, Chile + Colombia (CC-BY-SA-4.0)
    librispeech  LibriSpeech dev-clean English audiobooks (CC-BY-4.0)

  fresh blind test, never used for fitting, calibration or any decision (role "external2")
    Google crowdsourced Spanish, ARGENTINA (accent held out entirely)
    Multilingual LibriSpeech Spanish test (audiobooks; new source)
    People's Speech English test (archive.org speech; new source, CC-BY)

Run: python -m wi.add_v2
"""
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq

from .add_other import cache
from .config import EXT, MANIFEST, SEED

OUT = EXT / "v2"
HF = "https://huggingface.co/datasets"
GOOGLE = {"chile": "ylacombe/google-chilean-spanish", "colombia": "ylacombe/google-colombian-spanish",
          "argentina": "ylacombe/google-argentinian-spanish"}


def get(url, dest):
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(url, dest.with_suffix(".part"))
            dest.with_suffix(".part").rename(dest)
            return dest
        except Exception as e:
            print(f"retry {attempt + 1}/5 {url}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(url)


def tree(repo, path=""):
    import json
    return json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/tree/main/{path}?recursive=1"))


def by_speaker(df, spk, n, per):
    df = df.groupby(spk, group_keys=False).apply(lambda g: g.sample(min(per, len(g)), random_state=SEED))
    return df.sample(min(n, len(df)), random_state=SEED)


def google(country, n):
    repo = GOOGLE[country]
    files = sorted(x["path"] for x in tree(repo) if x["type"] == "file" and x["path"].endswith(".parquet"))
    pick = [next(f for f in files if f.startswith("female/")), next(f for f in files if f.startswith("male/"))]
    frames = []
    for f in pick:
        p = get(f"{HF}/{repo}/resolve/main/{f}", OUT / "google" / country / f.replace("/", "_"))
        frames.append(pq.read_table(p).to_pandas())
    t = pd.concat(frames, ignore_index=True)
    return by_speaker(t, "speaker_id", n, per=8)


def main():
    man = pd.read_parquet(MANIFEST)
    man = man[~man.corpus.isin(["google_es", "librispeech", "google_es_ar", "mls", "peoples_speech"])]
    rows, blobs = [], []

    def add(cid, corpus, label, role, group, lic, audio_bytes, ext_known=None):
        rows.append(dict(clip_id=cid, corpus=corpus, label=label, role=role, group=group, license=lic, src="",
                         start=None, dur=None, note="v2 data", os_split="", ext_known=ext_known,
                         role_orig="", exclude_reason=""))
        blobs.append(audio_bytes)

    for country in ["chile", "colombia"]:
        for r in google(country, 300).itertuples():
            add(f"gesp_{country}_{r.Index}", "google_es", "spa", "xcorpus", f"gesp:{country}:{r.speaker_id}",
                "cc-by-sa-4.0", r.audio["bytes"])
    for r in google("argentina", 300).itertuples():
        add(f"gesp_argentina_{r.Index}", "google_es_ar", "spa", "external2", f"gesp:argentina:{r.speaker_id}",
            "cc-by-sa-4.0", r.audio["bytes"], True)
    ls = pq.read_table(get(f"{HF}/openslr/librispeech_asr/resolve/main/all/validation.clean/0000.parquet",
                           OUT / "librispeech_dev_clean.parquet")).to_pandas()
    for r in by_speaker(ls, "speaker_id", 400, per=10).itertuples():
        add(f"ls_{r.id}", "librispeech", "eng", "xcorpus", f"ls:{r.speaker_id}", "cc-by-4.0", r.audio["bytes"])
    mls = pq.read_table(get(f"{HF}/facebook/multilingual_librispeech/resolve/main/spanish/test-00000-of-00001.parquet",
                            OUT / "mls_spanish_test.parquet")).to_pandas()
    spk = "speaker_id" if "speaker_id" in mls.columns else mls.columns[0]
    for r in by_speaker(mls, spk, 300, per=10).itertuples():
        add(f"mls_{r.Index}", "mls", "spa", "external2", f"mls:{getattr(r, spk)}", "cc-by-4.0", r.audio["bytes"], True)
    ps = pq.read_table(get(f"{HF}/MLCommons/peoples_speech/resolve/main/clean/test-00000-of-00009.parquet",
                           OUT / "peoples_speech_test0.parquet")).to_pandas()
    ps["item"] = ps["id"].astype(str).str.replace(r"_\d+\.flac$", "", regex=True)   # one archive.org recording
    for r in by_speaker(ps, "item", 300, per=5).itertuples():
        add(f"ps_{r.Index}", "peoples_speech", "eng", "external2", f"ps:{r.item}", "cc-by-2.0/2.5", r.audio["bytes"], True)

    with ThreadPoolExecutor(16) as pool:
        ok = list(pool.map(lambda rb: cache(rb[0]["clip_id"], rb[1]), zip(rows, blobs)))
    new = pd.DataFrame([r for r, good in zip(rows, ok) if good])
    man = pd.concat([man, new], ignore_index=True)
    man.to_parquet(MANIFEST)
    print(new.groupby(["role", "corpus", "label"]).agg(clips=("clip_id", "size"), groups=("group", "nunique")).to_string())


if __name__ == "__main__":
    main()
