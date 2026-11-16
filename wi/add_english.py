"""Add English as a 7th class ("no interpreter needed") without re-running the full prepare step.

Training clips come from Bloom-Speech's English *train* books; the English clips already in the
manifest (from Bloom's *test* books) become role "heldout_eng" and are used only for evaluation.
Appends rows to data/manifest.parquet and caches the audio.

Run: python -m wi.add_english
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .audio import decode
from .config import BLOOM_ENG, CACHE, MANIFEST, MAX_SECONDS, SEED, SR

CAP_PER_BOOK, TOTAL = 6, 1500


def main():
    man = pd.read_parquet(MANIFEST)
    if (man.corpus.eq("bloom") & man.label.eq("eng") & man.role.eq("train")).any():
        print("English training clips already added"); return
    man.loc[man.corpus.eq("bloom") & man.label.eq("eng"), "role"] = "heldout_eng"

    t = pq.read_table(BLOOM_ENG / "train" / "0000.parquet").to_pandas()
    t = t[t.license != "cc-by-nc-nd"]
    # Bloom's English train/test split is NOT book-disjoint (197 of 199 test books also appear in
    # train), so drop every book used by the held-out English clips.
    test_books = {g.split(":", 1)[1] for g in man[man.role == "heldout_eng"].group}
    t = t[~t.book.isin(test_books)]
    book_of = t.book.to_dict()
    t = (t.groupby("book", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), CAP_PER_BOOK), random_state=SEED),
                 include_groups=False))
    t = t.sample(min(TOTAL, len(t)), random_state=SEED)
    t["book"] = t.index.map(lambda i: book_of[i])
    rows = []
    for r in t.itertuples():
        cid = f"bloomengtr_{r.Index}"
        out = CACHE / f"{cid}.npy"
        if not out.exists():
            x = decode(r.audio["bytes"])[: int(MAX_SECONDS * SR)]
            if len(x) < int(0.5 * SR):
                continue
            np.save(out, x.astype(np.float16))
        rows.append(dict(clip_id=cid, corpus="bloom", label="eng", role="train",
                         group=f"bloom:{r.book}", license=r.license, src="", start=None, dur=None,
                         note="English train books"))
    man = pd.concat([man, pd.DataFrame(rows)], ignore_index=True)   # appended at the end
    man.to_parquet(MANIFEST)
    print(f"added {len(rows)} English training clips from {t.book.nunique()} books; "
          f"{(man.role == 'heldout_eng').sum()} English test clips held out")


if __name__ == "__main__":
    main()
