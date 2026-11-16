# Advanced methods

## A1 Several utterances from the same speaker

Clips from one recording (book, episode, lesson) are grouped into disjoint sets of n; evidence is combined by averaging log-probabilities.

| source | n=1 | n=2 | n=4 | n=8 | useful answer n=1 -> n=8 |
|---|---|---|---|---|---|
| Bloom held-out books | 77.9% | 81.9% | 83.6% | 84.5% | 76.7% -> 84.7% |
| coerll, Bloom-only model | 23.4% | 21.4% | 19.6% | 26.7% | 33.0% -> 42.2% |
| coerll, adapted model, held-out recordings | 83.3% | 95.3% | 100.0% | 100.0% | 79.5% -> 100.0% |
| tachiwin, Bloom-only model | 42.1% | 53.1% | 58.3% | 65.7% | 41.1% -> 23.9% |
| tachiwin, adapted model, held-out recordings | 91.6% | 96.1% | 98.8% | 100.0% | 95.1% -> 100.0% |
| tzijonel, Bloom-only model | 42.5% | 41.4% | 40.2% | 38.6% | 54.7% -> 52.3% |
| tzijonel, adapted model, held-out recordings | 91.8% | 98.7% | 100.0% | 100.0% | 91.1% -> 100.0% |

## A2 Erasing the recording corpus (LEACE), leave-one-corpus-out

| corpus left out | no erasure | random directions | naive LEACE | within-language LEACE | Bloom books macro-F1 (none -> within) | corpus probe AUROC (none -> within) |
|---|---|---|---|---|---|---|
| coerll | 8.9% | 8.4% | 8.2% | 8.9% | 0.740 -> 0.740 | 0.9915834194814237 -> 0.563242367982447 |
| tachiwin | 38.5% | 38.6% | 23.1% | 42.4% | 0.743 -> 0.747 | 0.9967129744480278 -> 0.7257920152905889 |
| tzijonel | 28.2% | 28.7% | 15.3% | 27.6% | 0.755 -> 0.756 | 0.9951108135054423 -> 0.71050200688945 |
| fleurs | 99.3% | 99.4% | 82.4% | 98.9% | 0.751 -> 0.753 | 0.9704047039048601 -> 0.2737793954559672 |
| google_es | 98.6% | 98.6% | 90.7% | 97.9% | 0.749 -> 0.749 | 0.9868500203028969 -> 0.5149969201259763 |
| librispeech | 100.0% | 100.0% | 100.0% | 100.0% | 0.751 -> 0.752 | 0.9912412439089083 -> 0.5401364605169348 |

### A2b Nuisance attribute projection (remove top-k between-recording directions within each language)

Sweep over k (not tuned on the left-out corpus). Accuracy on the corpus left out of training:

| corpus left out | k=0 | k=10 | k=25 | k=50 | k=100 | k=200 | k=random50 |
|---|---|---|---|---|---|---|---|
| coerll | 8.9% | 9.2% | 12.3% | 15.9% | 20.0% | 24.1% | 8.0% |
| tachiwin | 38.5% | 43.1% | 40.6% | 45.7% | 44.5% | 42.5% | 38.6% |
| tzijonel | 28.2% | 29.8% | 30.4% | 27.7% | 32.1% | 36.6% | 28.5% |
| fleurs | 99.3% | 99.6% | 99.6% | 95.9% | 99.5% | 99.3% | 99.3% |
| google_es | 98.6% | 95.4% | 97.9% | 97.1% | 96.4% | 93.2% | 98.6% |
| librispeech | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## A3 Mayan family tree

| setting | accuracy | branch accuracy | mean tree distance of errors | if errors were random |
|---|---|---|---|---|
| bloom_heldout_books_flat | 77.9% | 81.9% | 5.51 | 5.78 |
| bloom_heldout_books_hier | 75.7% | 80.3% | 5.36 | 5.78 |
| new_corpora_flat | 52.7% | 69.5% | 4.75 | 5.55 |
| new_corpora_hier | 52.7% | 67.6% | 4.93 | 5.55 |

Branch fallback threshold 0.97. Useful answers (right language, or a branch containing it): held-out Bloom books 65.0%; new corpora (Bloom-only model) coerll 18.1%, tachiwin 19.0%, tzijonel 50.5%

Languages it has no class for (shipped model):

| language | nearest covered relative | outcome shares |
|---|---|---|
| Tz'utujil | Kaqchikel, K'iche' | {'nearest branch': 0.61, 'said Mayan': 0.268, 'declined': 0.122} |
| Itza' | none close | {'said Mayan': 0.563, 'named unrelated language': 0.219, 'declined': 0.115, 'other branch': 0.104} |
| Poqomam | Kaqchikel, Q'eqchi', K'iche' | {'nearest branch': 0.386, 'said Mayan': 0.25, 'named nearest relative': 0.205, 'declined': 0.125, 'said Spanish/English': 0.019, 'other branch': 0.011, 'named unrelated language': 0.004} |
| Sipakapense | Kaqchikel, K'iche' | {'said Mayan': 0.365, 'nearest branch': 0.335, 'named nearest relative': 0.27, 'declined': 0.015, 'other branch': 0.01, 'named unrelated language': 0.005} |
| Sakapulteko | Kaqchikel, K'iche' | {'nearest branch': 0.548, 'said Mayan': 0.231, 'named nearest relative': 0.163, 'declined': 0.043, 'named unrelated language': 0.01, 'said Spanish/English': 0.005} |
| Popti' (Jakalteko) | Q'anjob'al | {'named nearest relative': 0.376, 'said Mayan': 0.319, 'nearest branch': 0.206, 'declined': 0.071, 'named unrelated language': 0.014, 'other branch': 0.014} |
| Yucatec Maya | none close | {'declined': 0.466, 'said Mayan': 0.358, 'other branch': 0.108, 'named unrelated language': 0.034, 'said Spanish/English': 0.034} |
| Tseltal | none close | {'declined': 0.567, 'named unrelated language': 0.22, 'said Mayan': 0.121, 'other branch': 0.092} |
| Tsotsil | none close | {'declined': 0.52, 'said Mayan': 0.236, 'named unrelated language': 0.128, 'other branch': 0.108, 'said Spanish/English': 0.007} |
| Ch'ol | none close | {'declined': 0.396, 'said Mayan': 0.289, 'named unrelated language': 0.188, 'other branch': 0.128} |
| Tojolabal | Q'anjob'al | {'said Mayan': 0.384, 'declined': 0.342, 'named unrelated language': 0.11, 'named nearest relative': 0.055, 'other branch': 0.055, 'nearest branch': 0.055} |
| Huastec | none close | {'said Mayan': 0.447, 'named unrelated language': 0.267, 'other branch': 0.147, 'declined': 0.113, 'said Spanish/English': 0.027} |
| Chuj | Q'anjob'al | {'said Mayan': 0.373, 'declined': 0.22, 'named nearest relative': 0.153, 'named unrelated language': 0.102, 'nearest branch': 0.068, 'other branch': 0.068, 'said Spanish/English': 0.017} |

Split-conformal 90% prediction sets (threshold 1.000): other half of Bloom books coverage 98.8%, mean set size 3.31.
New corpora, Bloom-only model: coerll 91.3% (size 6.02), tachiwin 93.4% (size 4.86), tzijonel 84.7% (size 4.37)
New corpora, held-out recordings, adapted model: coerll 100.0% (size 4.48), tachiwin 98.6% (size 2.42), tzijonel 100.0% (size 2.71)

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
