"""Record SHA-256 fingerprints of the shipped models BEFORE external scoring; wi.validate refuses to
score the external set if either file changed afterwards.

Run: python -m wi.freeze
"""
import hashlib
import json
import time

from .config import MODELS


def main():
    out = {name: hashlib.sha256((MODELS / name).read_bytes()).hexdigest()
           for name in ["router.joblib", "openset.joblib"]}
    out["frozen_at"] = time.strftime("%Y-%m-%d %H:%M %Z")
    out["backbones"] = ["facebook/mms-lid-4017 (frozen, layer 24)", "facebook/wav2vec2-xlsr-53-espeak-cv-ft (frozen)"]
    (MODELS / "FROZEN.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
