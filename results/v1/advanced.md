# Advanced methods

## A1 Several utterances from the same speaker

Clips from one recording (book, episode, lesson) are grouped into disjoint sets of n; evidence is combined by averaging log-probabilities.

| source | n=1 | n=2 | n=4 | n=8 | useful answer n=1 -> n=8 |
|---|---|---|---|---|---|
| Bloom held-out books | 77.9% | 81.9% | 83.6% | 84.5% | 76.7% -> 84.7% |
| coerll, Bloom-only model | 23.4% | 21.4% | 19.6% | 26.7% | 33.0% -> 42.2% |
| coerll, adapted model, held-out recordings | 83.7% | 95.3% | 100.0% | 100.0% | 79.8% -> 100.0% |
| tachiwin, Bloom-only model | 42.1% | 53.1% | 58.3% | 65.7% | 41.1% -> 23.9% |
| tachiwin, adapted model, held-out recordings | 91.6% | 96.1% | 98.8% | 100.0% | 95.1% -> 100.0% |
| tzijonel, Bloom-only model | 42.5% | 41.4% | 40.2% | 38.6% | 54.7% -> 52.3% |
| tzijonel, adapted model, held-out recordings | 91.8% | 98.7% | 100.0% | 100.0% | 91.3% -> 100.0% |

## A2 Erasing the recording corpus (LEACE), leave-one-corpus-out

| corpus left out | no erasure | random directions | naive LEACE | within-language LEACE | Bloom books macro-F1 (none -> within) | corpus probe AUROC (none -> within) |
|---|---|---|---|---|---|---|
| coerll | 8.4% | 8.4% | 7.5% | 8.4% | 0.739 -> 0.734 | 0.995228469235747 -> 0.5463516678257663 |
| tachiwin | 38.7% | 37.6% | 22.9% | 42.2% | 0.750 -> 0.749 | 0.9980735263366302 -> 0.7359900576755665 |
| tzijonel | 28.1% | 28.5% | 18.4% | 28.1% | 0.756 -> 0.753 | 0.9893478393378725 -> 0.7299159985041507 |
| fleurs | 94.6% | 93.6% | 94.8% | 93.9% | 0.748 -> 0.751 | 0.9590252410045357 -> 0.4492778282916904 |

### A2b Nuisance attribute projection (remove top-k between-recording directions within each language)

Sweep over k (not tuned on the left-out corpus). Accuracy on the corpus left out of training:

| corpus left out | k=0 | k=10 | k=25 | k=50 | k=100 | k=200 | k=random50 |
|---|---|---|---|---|---|---|---|
| coerll | 8.4% | 9.4% | 12.0% | 15.7% | 18.1% | 21.9% | 8.2% |
| tachiwin | 38.7% | 40.4% | 41.5% | 44.8% | 44.9% | 42.4% | 38.5% |
| tzijonel | 28.1% | 29.6% | 28.4% | 28.5% | 34.1% | 35.6% | 28.9% |
| fleurs | 94.6% | 95.7% | 98.5% | 94.4% | 95.7% | 98.8% | 94.1% |

## A3 Mayan family tree

| setting | accuracy | branch accuracy | mean tree distance of errors | if errors were random |
|---|---|---|---|---|
| bloom_heldout_books_flat | 77.9% | 81.9% | 5.51 | 5.78 |
| bloom_heldout_books_hier | 75.7% | 80.3% | 5.36 | 5.78 |
| new_corpora_flat | 39.0% | 60.7% | 4.75 | 5.55 |
| new_corpora_hier | 39.1% | 58.3% | 4.93 | 5.55 |

Branch fallback threshold 0.97. Useful answers (right language, or a branch containing it): held-out Bloom books 65.0%; new corpora (Bloom-only model) coerll 18.1%, tachiwin 19.0%, tzijonel 50.5%

Languages it has no class for (shipped model):

| language | nearest covered relative | outcome shares |
|---|---|---|
| Tz'utujil | Kaqchikel, K'iche' | {'nearest branch': 0.61, 'said Mayan': 0.244, 'declined': 0.146} |
| Itza' | none close | {'said Mayan': 0.568, 'named unrelated language': 0.213, 'declined': 0.115, 'other branch': 0.104} |
| Poqomam | Kaqchikel, Q'eqchi', K'iche' | {'nearest branch': 0.379, 'said Mayan': 0.246, 'named nearest relative': 0.201, 'declined': 0.136, 'said Spanish/English': 0.019, 'other branch': 0.015, 'named unrelated language': 0.004} |
| Sipakapense | Kaqchikel, K'iche' | {'said Mayan': 0.355, 'nearest branch': 0.34, 'named nearest relative': 0.265, 'declined': 0.025, 'other branch': 0.01, 'named unrelated language': 0.005} |
| Sakapulteko | Kaqchikel, K'iche' | {'nearest branch': 0.543, 'said Mayan': 0.24, 'named nearest relative': 0.159, 'declined': 0.043, 'named unrelated language': 0.01, 'said Spanish/English': 0.005} |
| Popti' (Jakalteko) | Q'anjob'al | {'named nearest relative': 0.355, 'said Mayan': 0.326, 'nearest branch': 0.22, 'declined': 0.064, 'named unrelated language': 0.014, 'other branch': 0.014, 'said Spanish/English': 0.007} |
| Yucatec Maya | none close | {'declined': 0.493, 'said Mayan': 0.338, 'other branch': 0.101, 'named unrelated language': 0.034, 'said Spanish/English': 0.034} |
| Tseltal | none close | {'declined': 0.574, 'named unrelated language': 0.234, 'said Mayan': 0.113, 'other branch': 0.078} |
| Tsotsil | none close | {'declined': 0.527, 'said Mayan': 0.236, 'named unrelated language': 0.128, 'other branch': 0.101, 'said Spanish/English': 0.007} |
| Ch'ol | none close | {'declined': 0.389, 'said Mayan': 0.302, 'named unrelated language': 0.188, 'other branch': 0.121} |
| Tojolabal | Q'anjob'al | {'said Mayan': 0.384, 'declined': 0.349, 'named unrelated language': 0.11, 'named nearest relative': 0.055, 'other branch': 0.055, 'nearest branch': 0.048} |
| Huastec | none close | {'said Mayan': 0.447, 'named unrelated language': 0.287, 'other branch': 0.127, 'declined': 0.113, 'said Spanish/English': 0.027} |
| Chuj | Q'anjob'al | {'said Mayan': 0.373, 'declined': 0.22, 'named nearest relative': 0.136, 'named unrelated language': 0.119, 'nearest branch': 0.085, 'other branch': 0.051, 'said Spanish/English': 0.017} |

Split-conformal 90% prediction sets (threshold 1.000): other half of Bloom books coverage 98.8%, mean set size 3.31.
New corpora, Bloom-only model: coerll 91.3% (size 6.02), tachiwin 93.4% (size 4.86), tzijonel 84.7% (size 4.37)
New corpora, held-out recordings, adapted model: coerll 100.0% (size 4.49), tachiwin 98.6% (size 2.46), tzijonel 100.0% (size 2.77)

## A4 Phonotactics (universal phone recogniser + phone n-grams)

Held-out Bloom books macro-F1: embedding 0.788, phones 0.655, fusion 0.802 (weight on embedding 0.5).

| new corpus | embedding | phones | fusion |
|---|---|---|---|
| coerll | 23.4% | 44.6% | 31.6% |
| tachiwin | 42.1% | 54.5% | 49.9% |
| tzijonel | 42.5% | 46.4% | 46.0% |

Corpus probe on phone n-grams (Bloom vs FLEURS Spanish): AUROC 0.988

Top phone n-grams per language:

- Spanish: ɾ · o s · e n · ð · β · e · s e · e s · l a · a s
- Mam: ŋ · b a x · i l · n t · b i l · l · t k · u n t · ʃ t · ʃ n
- K'iche': a r e · ʒ · p a l · l e · i k · k i · r i · w a tʃ · dʒ · k e
- Q'eqchi': l i · n a k · r a ʃ · k w · aː · s a l · l i k · i ŋ k · k · ʃ u l
- Kaqchikel: r u · x · ɬ · r i · i k · v i · n u · ç · iː · r u b
- Q'anjob'al: j · aɪ · tʃ i · tʃ a m · j e t · m i m · b a l · b aɪ · a ʃ · j e
- English: ə · z · æ · ɹ · ɪ · aɪ · ɚ · ð ə · ɐ · ʌ

Share of clips containing uvular /q/, glottal stop /ʔ/, /χ/: {"Spanish": {"q": 0.0, "ʔ": 0.0, "χ": 0.0}, "Mam": {"q": 0.12727272727272726, "ʔ": 0.0, "χ": 0.0}, "K'iche'": {"q": 0.23232323232323232, "ʔ": 0.0005941770647653001, "χ": 0.0}, "Q'eqchi'": {"q": 0.3276699029126214, "ʔ": 0.0, "χ": 0.0}, "Kaqchikel": {"q": 0.2606473594548552, "ʔ": 0.0, "χ": 0.0}, "Q'anjob'al": {"q": 0.10257680872150644, "ʔ": 0.0004955401387512388, "χ": 0.0}, "English": {"q": 0.0, "ʔ": 0.0, "χ": 0.0}}
