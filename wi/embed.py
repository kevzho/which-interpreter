"""Frozen speech embeddings from facebook/mms-lid-4017 (shared by feature extraction and the demo).

For each clip we keep:
  * the mean-pooled hidden state of transformer layer LID_LAYER (1280-d): input to our router
  * the model's own 4017-way language-ID logits: used for the zero-shot baseline
  * MFCC mean/std (+deltas, 160-d): a non-pretrained acoustic baseline
"""
import numpy as np
import torch

from .config import LID_LAYER, LID_MODEL, SR

ZS_CODES = ["spa", "mam", "quc", "kek", "cak", "kjb"]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    def __init__(self, device=None):
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
        self.device = device or pick_device()
        self.dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32
        self.fe = AutoFeatureExtractor.from_pretrained(LID_MODEL)
        # load straight into half precision: avoids a ~4 GB fp32 copy at start-up
        self.model = (Wav2Vec2ForSequenceClassification.from_pretrained(LID_MODEL, dtype=self.dtype)
                      .to(self.device).eval())
        self.full = True
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        l2i = {v: k for k, v in self.id2label.items()}
        self.zs_index = [l2i[c] for c in ZS_CODES]

    def truncate(self):
        """Keep only the layers needed for hidden_states[LID_LAYER] (~2x faster, no logits).
        With stable layer norm, hidden_states[k] is the input to layer k+1, so keeping LID_LAYER+1
        layers returns exactly the same layer-LID_LAYER state as the full model."""
        enc = self.model.wav2vec2.encoder
        enc.layers = enc.layers[: LID_LAYER + 1]
        self.full = False

    def free(self):
        if self.device == "mps":
            torch.mps.empty_cache()

    @torch.no_grad()
    def __call__(self, waves):
        """waves: list of 1-D float arrays at 16 kHz. Returns (hidden[N,1280], logits[N,4017] or None)."""
        waves = [np.pad(w, (0, max(0, 8000 - len(w)))).astype(np.float32) for w in waves]
        # pad every batch to a multiple of 2 s so the GPU sees only a few distinct shapes
        # (the MPS allocator caches one buffer set per shape; arbitrary lengths make memory explode)
        bucket = 2 * SR
        target = int(np.ceil(max(len(w) for w in waves) / bucket) * bucket)
        inp = self.fe(waves, sampling_rate=SR, return_tensors="pt", padding="max_length",
                      max_length=target, return_attention_mask=True)
        iv = inp["input_values"].to(self.device, dtype=self.dtype)
        am = inp["attention_mask"].to(self.device)
        out = self.model(iv, attention_mask=am, output_hidden_states=True)
        h = out.hidden_states[LID_LAYER]
        fmask = self.model.wav2vec2._get_feature_vector_attention_mask(h.shape[1], am)
        fmask = fmask.unsqueeze(-1).to(h.dtype)
        pooled = (h * fmask).sum(1) / fmask.sum(1).clamp(min=1)
        logits = out.logits.float().cpu().numpy() if self.full else None
        pooled = pooled.float().cpu().numpy()
        del out, h, iv, am
        return pooled, logits


def mfcc_stats(x):
    import torchaudio
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]
    m = torchaudio.transforms.MFCC(sample_rate=SR, n_mfcc=40,
                                   melkwargs=dict(n_fft=400, hop_length=160, n_mels=64))(t)[0]
    d = torchaudio.functional.compute_deltas(m[None])[0]
    f = torch.cat([m, d])
    return torch.cat([f.mean(1), f.std(1)]).numpy()
