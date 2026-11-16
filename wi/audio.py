"""Audio decoding, long-recording segmentation and augmentation (noise, reverb, phone line)."""
import subprocess
from functools import lru_cache

import numpy as np
import soundfile as sf

from .config import SR, NOISE, SEED


def decode(src, start=None, dur=None, sr=SR):
    """Decode a file path or raw bytes to mono float32 at `sr` using ffmpeg."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    if isinstance(src, (bytes, bytearray)):
        cmd += ["-i", "pipe:0"]
        inp = bytes(src)
    else:
        cmd += ["-i", str(src)]
        inp = None
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    out = subprocess.run(cmd, input=inp, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def speech_windows(x, win_s, max_windows, skip_s=30.0, min_speech=0.6, sr=SR):
    """Pick up to `max_windows` evenly spread windows that are mostly non-silent.

    Energy-based: a 25 ms frame counts as speech if it is within 30 dB of the recording's
    95th-percentile frame energy. Skips `skip_s` seconds at both ends (intros, music)."""
    hop = int(0.025 * sr)
    n = len(x) // hop
    if n == 0:
        return []
    e = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
    voiced = e > np.percentile(e, 95) - 30
    win = int(win_s * sr)
    skip = int(skip_s * sr) if len(x) > 4 * int(skip_s * sr) else 0
    starts = range(skip, max(skip, len(x) - skip - win), win)
    good = [s for s in starts if voiced[s // hop:(s + win) // hop].mean() >= min_speech]
    if len(good) > max_windows:
        idx = np.linspace(0, len(good) - 1, max_windows).round().astype(int)
        good = [good[i] for i in idx]
    return [(s / sr, win_s) for s in good]


# ---------------------------------------------------------------- augmentation
# Training augmentation uses the first slice of each shuffled file list; the test views (noise /
# reverb robustness tests) use a disjoint second slice, so no test noise or room is seen in training.
POOL_SLICES = {"train": (0, 400), "test": (400, 800)}
RIR_SLICES = {"train": (0, 300), "test": (300, 600)}


@lru_cache(maxsize=2)
def _noise_pool(split="train"):
    """Load short noise excerpts (DEMAND, ESC-50, RIRS point-source noises) at 16 kHz."""
    # same (unsorted) order as the original augmentation run, so the "train" slice is exactly the
    # set of files the existing aug features were made with and the "test" slice is disjoint from it
    files = [p for p in NOISE.rglob("*.wav")
             if "rir" not in p.name.lower() and "simulated_rirs" not in str(p)]
    rng = np.random.default_rng(SEED)
    rng.shuffle(files)
    a, b = POOL_SLICES[split]
    pool = []
    for p in files[a:b]:
        try:
            pool.append(decode(p, dur=10.0))
        except Exception:
            pass
    return [n for n in pool if len(n) > SR and np.std(n) > 1e-4]


@lru_cache(maxsize=2)
def _rir_pool(split="train"):
    files = sorted((NOISE).rglob("simulated_rirs/**/*.wav"))
    rng = np.random.default_rng(SEED)
    rng.shuffle(files)
    a, b = RIR_SLICES[split]
    out = []
    for p in files[a:b]:
        h, r = sf.read(p, dtype="float32")
        h = h if h.ndim == 1 else h[:, 0]
        if r == SR:
            out.append(h / (np.abs(h).max() + 1e-9))
    return out


def add_noise(x, rng, snr_db, split="train"):
    pool = _noise_pool(split)
    n = pool[rng.integers(len(pool))]
    if len(n) < len(x):
        n = np.tile(n, int(np.ceil(len(x) / len(n))))
    off = rng.integers(0, len(n) - len(x) + 1)
    n = n[off:off + len(x)]
    ps, pn = np.mean(x ** 2) + 1e-10, np.mean(n ** 2) + 1e-10
    return x + n * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))


def reverb(x, rng, split="train"):
    pool = _rir_pool(split)
    h = pool[rng.integers(len(pool))]
    from scipy.signal import fftconvolve      # direct convolution is ~100x slower for 1 s RIRs
    y = fftconvolve(x, h)[: len(x)]
    return y * (np.abs(x).max() / (np.abs(y).max() + 1e-9))


def phone(x):
    """Narrow-band phone line: 8 kHz, 300-3400 Hz band, 8-bit mu-law, back to 16 kHz."""
    import torch
    import torchaudio.functional as F
    t = torch.from_numpy(x.astype(np.float32))[None]
    t = F.resample(t, SR, 8000)
    t = F.highpass_biquad(t, 8000, 300.0)
    t = F.lowpass_biquad(t, 8000, 3400.0)
    t = t / (t.abs().max() + 1e-9) * 0.9
    t = F.mu_law_decoding(F.mu_law_encoding(t, 256), 256)
    t = F.resample(t, 8000, SR)
    return t[0].numpy()


def augment(x, kind, rng):
    if kind == "noise":                 # test view: held-out noise files
        return add_noise(x, rng, snr_db=10.0, split="test")
    if kind == "reverb":                # test view: held-out simulated rooms
        return reverb(x, rng, split="test")
    if kind == "phone":
        return phone(x)
    if kind == "random":   # training-time augmentation: random combination
        y = x
        if rng.random() < 0.5:
            y = reverb(y, rng)
        if rng.random() < 0.7:
            y = add_noise(y, rng, snr_db=float(rng.uniform(5, 20)))
        if rng.random() < 0.3:
            y = phone(y)
        return y
    raise ValueError(kind)
