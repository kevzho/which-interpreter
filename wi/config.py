"""Paths and constants shared by every pipeline stage."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BLOOM = DATA / "bloom"
EXT = DATA / "external"
BLOOM_ENG = ROOT / "bloom-speech-parquet" / "eng"
CACHE = DATA / "cache" / "wav"          # 16 kHz float16 clips, one .npy per clip
FEATS = DATA / "features"               # embeddings per "view" (clean, aug, test_noise, ...)
NOISE = EXT / "noise" / "extracted"
MANIFEST = DATA / "manifest.parquet"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

SR = 16000
MAX_SECONDS = 8.0                       # clips are truncated to this before embedding
WINDOW_SECONDS = 6.0                    # window length when cutting long recordings
SEED = 0

# Classes the router can answer with. Everything else should be abstained on.
# English is a class so an English speaker gets "no interpreter needed" instead of "Spanish".
LANGS = ["spa", "mam", "quc", "kek", "cak", "kjb", "eng"]
MAYAN = ["mam", "quc", "kek", "cak", "kjb"]
NAMES = {
    "spa": "Spanish", "mam": "Mam", "quc": "K'iche'", "kek": "Q'eqchi'",
    "cak": "Kaqchikel", "kjb": "Q'anjob'al", "tzj": "Tz'utujil", "eng": "English",
    "itz": "Itza'", "poc": "Poqomam", "qum": "Sipakapense", "quv": "Sakapulteko",
    "cac": "Chuj", "jac": "Popti' (Jakalteko)", "yua": "Yucatec Maya", "tzh": "Tseltal",
    "tzo": "Tsotsil", "ctu": "Ch'ol", "toj": "Tojolabal", "hus": "Huastec", "cab": "Garifuna",
    "lac": "Lacandon", "fra": "French", "por": "Portuguese",
}

LID_MODEL = "facebook/mms-lid-4017"
LID_LAYER = 24                          # best layer in the Nov 14 probe (book-grouped macro-F1 0.86)
