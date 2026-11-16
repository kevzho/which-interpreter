"""More varied "other language" data from Google FLEURS (CC-BY-4.0): many speakers and microphones,
closer to live microphone speech than storybook recordings.

* ~40 non-target languages (validation split, 80 clips each) -> role "other". A language that
  already exists in the manifest keeps its os_split (e.g. French stays a TEST language); new
  languages get a seeded 60/20/20 train/tune/test split by language.
* English (en_us): validation speakers -> extra training for the English class ("train_extra");
  test speakers -> held-out English ("xcorpus"). FLEURS splits are speaker-disjoint.

Run: python -m wi.add_fleurs
"""
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .add_other import cache, split_units
from .config import EXT, MANIFEST, SEED

LANGS = {"fr_fr": "fra", "it_it": "ita", "pt_br": "por", "ca_es": "cat", "ro_ro": "ron", "gl_es": "glg",
         "oc_fr": "oci", "ast_es": "ast", "de_de": "deu", "nl_nl": "nld", "sv_se": "swe", "da_dk": "dan",
         "ru_ru": "rus", "pl_pl": "pol", "uk_ua": "ukr", "cs_cz": "ces", "el_gr": "ell", "hu_hu": "hun",
         "fi_fi": "fin", "tr_tr": "tur", "cmn_hans_cn": "cmn", "yue_hant_hk": "yue", "ja_jp": "jpn",
         "ko_kr": "kor", "vi_vn": "vie", "th_th": "tha", "id_id": "ind", "fil_ph": "tgl", "hi_in": "hin",
         "ta_in": "tam", "bn_in": "ben", "ur_pk": "urd", "ar_eg": "arb", "fa_ir": "pes", "he_il": "heb",
         "sw_ke": "swh", "yo_ng": "yor", "ha_ng": "hau", "am_et": "amh", "zu_za": "zul"}
PER_LANG = 80
OUT = EXT / "fleurs_other"


def fetch(config, split):
    f = OUT / f"{config}_{split}.parquet"
    if not f.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/api/datasets/google/fleurs/parquet/{config}/{split}/0.parquet"
        for attempt in range(5):          # the HF CDN occasionally drops connections
            try:
                urllib.request.urlretrieve(url, f.with_suffix(".part"))
                f.with_suffix(".part").rename(f)
                break
            except Exception as e:
                print(f"retry {attempt + 1}/5 {config} {split}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"could not download {config} {split}")
    return f


def main():
    man = pd.read_parquet(MANIFEST)
    man = man[~man.clip_id.str.startswith(("fleursother_", "fleurseng_"))]   # idempotent
    with ThreadPoolExecutor(6) as pool:
        files = dict(zip(LANGS, pool.map(lambda c: fetch(c, "validation"), LANGS)))
        eng_files = list(pool.map(lambda s: fetch("en_us", s), ["validation", "test"]))
    existing = man[man.os_split.fillna("") != ""].groupby("label").os_split.first().to_dict()
    new_split = split_units([iso for iso in LANGS.values() if iso not in existing], seed=SEED + 7)
    rows, blobs = [], []
    for config, iso in LANGS.items():
        t = pq.read_table(files[config], columns=["id", "audio"]).to_pandas()
        t = t.drop_duplicates("id").sample(min(PER_LANG, len(t)), random_state=SEED)   # id = sentence
        for r in t.itertuples():
            rows.append(dict(clip_id=f"fleursother_{iso}_{r.Index}", corpus="fleurs", label=iso, role="other",
                             group=f"fleurs:{iso}:{r.id}", license="cc-by-4.0", src="", start=None, dur=None,
                             note=f"FLEURS {config}", os_split=existing.get(iso, new_split.get(iso))))
            blobs.append(r.audio["bytes"])
    for f, role, n in [(eng_files[0], "train_extra", 400), (eng_files[1], "xcorpus", 300)]:
        t = pq.read_table(f, columns=["id", "audio"]).to_pandas()
        t = t.sample(min(n, len(t)), random_state=SEED)
        for r in t.itertuples():
            rows.append(dict(clip_id=f"fleurseng_{role}_{r.Index}", corpus="fleurs", label="eng", role=role,
                             group=f"fleurs:eng:{r.id}", license="cc-by-4.0", src="", start=None, dur=None,
                             note="FLEURS en_us", os_split=""))
            blobs.append(r.audio["bytes"])
    with ThreadPoolExecutor(16) as pool:
        ok = list(pool.map(lambda rb: cache(rb[0]["clip_id"], rb[1]), zip(rows, blobs)))
    new = pd.DataFrame([r for r, good in zip(rows, ok) if good])
    man = pd.concat([man, new], ignore_index=True)
    man.to_parquet(MANIFEST)
    print(new.groupby(["role", "os_split"]).agg(clips=("clip_id", "size"), languages=("label", "nunique")).to_string())
    print("FLEURS other-language splits:", new[new.role == "other"].groupby("os_split").label.unique().map(sorted).to_dict())


if __name__ == "__main__":
    main()
