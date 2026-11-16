"""Phone transcription with a universal phoneme recogniser (idea 4: phonotactic language ID).

facebook/wav2vec2-xlsr-53-espeak-cv-ft outputs IPA phones for any language. We decode greedily
(argmax per frame, collapse repeats, drop blanks) straight from its vocab, so the `phonemizer`
package is not needed. The phone strings feed an n-gram classifier in wi.advanced.

Output: data/features/phones.parquet (clip_id, phones) - resumable in chunks like wi.features.

Run: python -m wi.phones
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from .config import CACHE, FEATS, MANIFEST, SR
from .embed import pick_device
from .features import CHUNK, batches

PHONE_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"


class PhoneRecognizer:
    def __init__(self, device=None):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        self.device = device or pick_device()
        self.dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(PHONE_MODEL)
        self.model = (Wav2Vec2ForCTC.from_pretrained(PHONE_MODEL, dtype=self.dtype)
                      .to(self.device).eval())
        vocab = json.load(open(hf_hub_download(PHONE_MODEL, "vocab.json")))
        self.id2tok = {v: k for k, v in vocab.items()}
        self.blank = self.model.config.pad_token_id
        self.skip = {self.blank} | {v for k, v in vocab.items() if k.startswith("<") or k == "|"}

    @torch.no_grad()
    def __call__(self, waves):
        waves = [np.pad(w, (0, max(0, 8000 - len(w)))).astype(np.float32) for w in waves]
        target = int(np.ceil(max(len(w) for w in waves) / (2 * SR)) * 2 * SR)
        inp = self.fe(waves, sampling_rate=SR, return_tensors="pt", padding="max_length",
                      max_length=target, return_attention_mask=True)
        logits = self.model(inp["input_values"].to(self.device, dtype=self.dtype),
                            attention_mask=inp["attention_mask"].to(self.device)).logits
        nframes = self.model._get_feat_extract_output_lengths(inp["attention_mask"].sum(1))
        ids = logits.argmax(-1).cpu().numpy()
        out = []
        for row, n in zip(ids, nframes.tolist()):
            seq, prev = [], None
            for t in row[:n]:
                if t != prev and t not in self.skip:
                    seq.append(self.id2tok[int(t)])
                prev = t
            out.append(" ".join(seq))
        return out

    def free(self):
        if self.device == "mps":
            torch.mps.empty_cache()


def main():
    man = pd.read_parquet(MANIFEST)
    man = man[man.role != "demo"].reset_index(drop=True)
    out_dir = FEATS / "phones"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    existing = sorted(out_dir.glob("chunk_*.parquet"))
    for p in existing:
        done |= set(pd.read_parquet(p).clip_id)
    todo = man[~man.clip_id.isin(done)]
    rec = PhoneRecognizer()
    t0, nch = time.time(), (len(todo) + CHUNK - 1) // CHUNK
    for c in range(nch):
        part = todo.iloc[c * CHUNK:(c + 1) * CHUNK]
        with ThreadPoolExecutor(4) as pool:
            waves = list(pool.map(lambda cid: np.load(CACHE / f"{cid}.npy").astype(np.float32),
                                  part.clip_id))
        order = np.argsort([len(w) for w in waves])
        phones = [""] * len(waves)
        for idx in batches(order, [len(w) for w in waves]):
            for j, s in zip(idx, rec([waves[j] for j in idx])):
                phones[j] = s
        pd.DataFrame({"clip_id": part.clip_id.values, "phones": phones}).to_parquet(
            out_dir / f"chunk_{len(existing) + c:04d}.parquet")
        rec.free()
        eta = (time.time() - t0) / (c + 1) * (nch - c - 1) / 60
        print(f"[phones] chunk {c + 1}/{nch}  eta {eta:.0f} min", flush=True)
    allp = pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("chunk_*.parquet"))])
    allp.to_parquet(FEATS / "phones.parquet")
    print(f"[phones] merged {len(allp)} clips")


if __name__ == "__main__":
    main()
