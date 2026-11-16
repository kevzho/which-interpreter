PY := .venv/bin/python
SHELL := /bin/bash   # recipes use `set -o pipefail` so a failing stage stops the chain despite tee

# Memory caps: RAM per process (wi/memcap.py watchdog), GPU via the MPS allocator watermarks,
# and maths-library threads (fewer buffers, leaves CPU for other work). Override e.g. `make WI_MEM_CAP_GB=4 ...`
export WI_MEM_CAP_GB ?= 6
export PYTORCH_MPS_HIGH_WATERMARK_RATIO ?= 0.4
export PYTORCH_MPS_LOW_WATERMARK_RATIO ?= 0.3
export OMP_NUM_THREADS ?= 4
export VECLIB_MAXIMUM_THREADS ?= 4
export OPENBLAS_NUM_THREADS ?= 4

.PHONY: synthetic ownvoice compare all download prepare other fleurs external v2 hygiene neardup freeze validate features phones experiments advanced ship openset app

all: prepare other fleurs external v2 hygiene features phones neardup experiments advanced ship openset freeze validate compare

download:            ## open-licence corpora (Bloom-Speech needs the HF click-through first)
	cd data/bloom && $(PY) download_bloom.py && $(PY) clean_manifest.py
	data/external/download_open.sh

prepare:             ## manifest over all corpora + 16 kHz clip cache
	set -o pipefail; $(PY) -u -m wi.prepare | tee logs/prepare.log

other:               ## add other-language clips + language-disjoint open-set splits (after prepare)
	set -o pipefail; $(PY) -u -m wi.add_other | tee logs/add_other.log

fleurs:              ## FLEURS: ~40 other languages (many speakers/mics) + extra English (after other)
	set -o pipefail; $(PY) -u -m wi.add_fleurs | tee logs/add_fleurs.log

features:            ## MMS-LID embeddings for clean, augmented and degraded views (resumable)
	set -o pipefail; $(PY) -u -m wi.features | tee logs/features.log

phones:              ## universal phone recogniser -> IPA strings for every clip (resumable)
	set -o pipefail; $(PY) -u -m wi.phones | tee logs/phones.log

experiments:         ## all evaluations, figures, final model
	set -o pipefail; $(PY) -u -m wi.experiments | tee logs/experiments.log

advanced:            ## A1 several utterances, A2 LEACE/NAP, A3 family tree + conformal, A4 phonotactics
	set -o pipefail; $(PY) -u -m wi.advanced | tee logs/advanced.log

ship:                ## final demo model: embedding + phone fusion, adapted; scored on held-out data
	set -o pipefail; $(PY) -u -m wi.ship | tee logs/ship.log

external:            ## frozen external validation sources (VoxPopuli, CIEMPIESS, EdAcc, Lingua Libre, Wikitongues)
	set -o pipefail; $(PY) -u -m wi.add_external | tee logs/add_external.log

v2:                  ## extra Spanish/English sources (speaker-split) + fresh blind test set (after external)
	set -o pipefail; $(PY) -u -m wi.add_v2 | tee logs/add_v2.log

hygiene:             ## drop duplicate/jingle audio, merge re-uploaded episodes, fix groups (after add_* steps)
	set -o pipefail; $(PY) -u -m wi.hygiene | tee logs/hygiene.log

neardup:             ## drop near-duplicate audio across recordings (re-encoded jingles/intros); after features
	set -o pipefail; $(PY) -u -m wi.neardup | tee logs/neardup.log

freeze:              ## fingerprint the shipped models before external scoring
	$(PY) -m wi.freeze

validate:            ## score the frozen shipped models once on the external set; bootstrap CIs
	set -o pipefail; $(PY) -u -m wi.validate | tee logs/validate.log

compare:             ## v1 vs v2 frozen models on the same clips; side-by-side examples
	$(PY) -m wi.compare_versions && $(PY) -m wi.showcase

openset:             ## "I don't know": outlier exposure + distance scores, language-disjoint protocol
	set -o pipefail; $(PY) -u -m wi.openset | tee logs/openset.log

app:                 ## live demo
	$(PY) app.py

synthetic:           ## text-to-speech voices in 5 languages through the frozen pipeline (macOS)
	set -o pipefail; $(PY) -u -m wi.synthetic_voices | tee logs/synthetic_voices.log

ownvoice:            ## score own-voice recordings saved from the demo (frozen models)
	set -o pipefail; $(PY) -u -m wi.own_voice | tee logs/own_voice.log
