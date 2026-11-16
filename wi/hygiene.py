"""Data hygiene found by the Nov 15 leakage audit. Run after all add_* steps, before features.

0. Silent clips (all-zero cached audio) are excluded.
1. Tzijonel: episodes uploaded twice (folder suffix _YYYYMM) become ONE group, so both copies land on
   the same side of the 50/50 new-corpus split.
2. Tzijonel: the 'Sbk-episode25 ... Sailing Beyond Knowledge' folder is an English-language podcast
   labelled Kaqchikel -> excluded.
3. Tachiwin: clips with no WOAF source URL were all grouped as one placeholder -> group by the track
   title (TIT2), or by clip if that is missing too.
4. Identical audio (SHA-1 of the cached 16 kHz clip) that occurs in more than one group or under more
   than one label is an intro, jingle or announcer insert, not evidence about one language -> every
   copy is excluded. Identical copies inside one group keep the first copy only.
5. Wikitongues K'iche' (Lorenzo) was missed because the file name is URL-encoded -> its windows are
   added to the external validation set.

Excluded rows get role "excluded" (original role kept in role_orig, reason in exclude_reason), so no
training, tuning or evaluation code reads them.
Run: python -m wi.hygiene
"""
import hashlib
import re
import urllib.parse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .audio import decode, speech_windows
from .config import CACHE, EXT, MANIFEST, MAX_SECONDS, SR, WINDOW_SECONDS


def clip_hash(cid):
    f = CACHE / f"{cid}.npy"
    return hashlib.sha1(np.load(f).tobytes()).hexdigest() if f.exists() else None


def exclude(man, mask, reason):
    mask = mask & (man.role != "excluded")
    man.loc[mask, "role_orig"] = man.loc[mask, "role"]
    man.loc[mask, "exclude_reason"] = reason
    man.loc[mask, "role"] = "excluded"
    return int(mask.sum())


def main():
    man = pd.read_parquet(MANIFEST)
    for col in ["role_orig", "exclude_reason"]:
        if col not in man:
            man[col] = ""
    log = {}

    # 1-2 Tzijonel
    tz = man.corpus.eq("tzijonel")
    man.loc[tz, "group"] = man.loc[tz, "group"].map(lambda g: re.sub(r"_20\d{4}$", "", g))
    log["tzijonel English podcast excluded"] = exclude(man, tz & man.group.str.contains("Sbk-episode25"),
                                                       "English-language podcast mislabelled as Kaqchikel")

    # 3 Tachiwin placeholder groups
    bad = man.corpus.eq("tachiwin") & man.group.isin(["tach:None", "tach:", "tach:nan"])
    if bad.any():
        titles = {}
        for f in sorted((EXT / "tachiwin" / "data").glob("*.parquet")):
            t = pq.read_table(f, columns=["id", "TIT2"]).to_pandas()
            titles.update(dict(zip(t.id.astype(str), t.TIT2)))
        def regroup(cid):
            raw = cid.split("_", 1)[1]
            title = titles.get(raw)
            return f"tach:title:{title}" if isinstance(title, str) and title else f"tach:clip:{raw}"
        man.loc[bad, "group"] = man.loc[bad, "clip_id"].map(regroup)
    log["tachiwin placeholder groups fixed"] = int(bad.sum())

    # 0 silent clips (all-zero first 8 s: FLEURS leading silence or failed decodes). They were labelled
    # Spanish and taught the model "silence = Spanish"; they are not evidence about any language.
    def peak(cid):
        f = CACHE / f"{cid}.npy"
        return float(np.abs(np.load(f).astype(np.float32)).max()) if f.exists() else 1.0
    silent = man.clip_id.map(peak) < 1e-3
    log["silent clips excluded"] = exclude(man, silent, "silent audio (no speech in the first 8 s)")
    relabel = silent & man.exclude_reason.fillna("").str.startswith("near-duplicate")
    man.loc[relabel, "exclude_reason"] = "silent audio (no speech in the first 8 s)"
    log["silent clips previously excluded as near-duplicates (reason corrected)"] = int(relabel.sum())

    # 4 identical audio
    live = man.role.ne("excluded")
    man.loc[live, "audio_sha1"] = man.loc[live, "clip_id"].map(clip_hash)
    h = man[live & man.audio_sha1.notna()]
    stats = h.groupby("audio_sha1").agg(n=("clip_id", "size"), groups=("group", "nunique"),
                                        labels=("label", "nunique"))
    cross = stats[(stats.groups > 1) | (stats.labels > 1)].index
    log["identical audio across groups/labels excluded"] = exclude(
        man, man.audio_sha1.isin(cross), "identical audio in more than one recording or label (intro/jingle)")
    same = stats[(stats.n > 1) & (stats.groups == 1) & (stats.labels == 1)].index
    dup_within = man.audio_sha1.isin(same) & man.duplicated("audio_sha1", keep="first") & man.role.ne("excluded")
    log["duplicate copies within one group excluded"] = exclude(man, dup_within, "duplicate copy within one recording")

    # 5 Wikitongues K'iche'
    if not man.clip_id.str.startswith("wt_quc_").any():
        added = 0
        for p in sorted((EXT / "wikitongues").glob("*")):
            name = urllib.parse.unquote(p.name.split("?")[0])
            if "K'iche'" not in name:
                continue
            x = decode(p)
            for k, (s, d) in enumerate(speech_windows(x, WINDOW_SECONDS, 8)):
                cid = f"wt_quc_Lorenzo_{k}"
                seg = x[int(s * SR): int((s + min(d, MAX_SECONDS)) * SR)]
                np.save(CACHE / f"{cid}.npy", seg.astype(np.float16))
                man = pd.concat([man, pd.DataFrame([dict(
                    clip_id=cid, corpus="wikitongues", label="quc", role="external", group=f"wt:{name}",
                    license="cc-by-sa", src=str(p), start=s, dur=d, note="external validation",
                    os_split="", ext_known=True, role_orig="", exclude_reason="")])], ignore_index=True)
                added += 1
        log["wikitongues K'iche' windows added"] = added

    man.to_parquet(MANIFEST)
    for k, v in log.items():
        print(f"{k}: {v}")
    ex = man[man.role == "excluded"]
    print(ex.groupby(["role_orig", "corpus", "exclude_reason"]).size().to_string())


if __name__ == "__main__":
    main()
