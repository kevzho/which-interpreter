# Results

## E1 Bloom-Speech, held-out books

| model | macro-F1 | accuracy | Mayan sent to Spanish | Spanish sent to Mayan |
|---|---|---|---|---|
| baseline_always_spanish | 0.050 | 21.4% | 100.0% | 0.0% |
| baseline_mms_lid_zero_shot | 0.548 | 63.3% | 23.7% | 1.0% |
| baseline_mfcc | 0.489 | 51.7% | 9.4% | 11.7% |
| bloom | 0.796 | 79.0% | 3.9% | 2.4% |
| bloom+aug | 0.789 | 78.1% | 4.2% | 3.0% |
| bloom+aug+fleurs | 0.788 | 77.9% | 4.0% | 3.1% |
| leaky_random_split | 0.977 | 97.7% | 0.6% | 1.9% |

Leakage gap (random split minus held-out books): 0.182
Book-bootstrap 95% CI for bloom+aug+fleurs: [0.6427435529361705, 0.8615443905588134]
Per-language recall (bloom+aug+fleurs): Spanish 96.7%, Mam 46.0%, K'iche' 56.2%, Q'eqchi' 94.8%, Kaqchikel 79.8%, Q'anjob'al 89.7%, English 97.5%
Abstention threshold 0.99: {'coverage': 0.5602770885028949, 'accuracy_on_answered': 0.9647536445838716, 'mayan_sent_to_spanish_answered': 0.012593703148425787}

## E2 Corpora never seen in training

| condition | corpus | language | n | always-Spanish | MMS zero-shot | MFCC | bloom | bloom+aug | bloom+aug+fleurs | answered at threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | coerll | K'iche' | 415 | 0.0% | 24.3% | 2.9% | 12.0% | 22.2% | 23.4% | 5.8% |
| clean | fleurs | English | 300 | 0.0% | 0.0% | 9.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| clean | fleurs | Spanish | 1316 | 100.0% | 100.0% | 14.8% | 100.0% | 100.0% | 100.0% | 99.6% |
| clean | google_es | Spanish | 280 | 100.0% | 99.6% | 29.6% | 99.6% | 99.6% | 99.6% | 90.0% |
| clean | librispeech | English | 400 | 0.0% | 0.0% | 41.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| clean | tachiwin | Mam | 715 | 0.0% | 39.0% | 4.5% | 42.1% | 39.7% | 42.1% | 23.2% |
| clean | tzijonel | Kaqchikel | 1219 | 0.0% | 49.8% | 3.4% | 39.7% | 42.3% | 42.5% | 26.7% |
| test_noise | coerll | K'iche' | 359 |  |  |  | 30.4% | 31.5% | 32.9% | 7.5% |
| test_noise | fleurs | English | 300 |  |  |  | 100.0% | 100.0% | 100.0% | 100.0% |
| test_noise | fleurs | Spanish | 300 |  |  |  | 100.0% | 100.0% | 100.0% | 99.7% |
| test_noise | google_es | Spanish | 280 |  |  |  | 99.6% | 99.3% | 99.6% | 91.1% |
| test_noise | librispeech | English | 300 |  |  |  | 99.3% | 99.7% | 99.7% | 99.3% |
| test_noise | tachiwin | Mam | 457 |  |  |  | 53.2% | 43.5% | 44.0% | 24.3% |
| test_noise | tzijonel | Kaqchikel | 504 |  |  |  | 36.3% | 38.5% | 38.7% | 24.6% |
| test_reverb | coerll | K'iche' | 359 |  |  |  | 20.1% | 21.7% | 23.1% | 4.7% |
| test_reverb | fleurs | English | 300 |  |  |  | 100.0% | 100.0% | 100.0% | 99.7% |
| test_reverb | fleurs | Spanish | 300 |  |  |  | 99.7% | 100.0% | 100.0% | 98.3% |
| test_reverb | google_es | Spanish | 280 |  |  |  | 97.9% | 96.4% | 97.1% | 67.1% |
| test_reverb | librispeech | English | 300 |  |  |  | 99.7% | 100.0% | 100.0% | 99.3% |
| test_reverb | tachiwin | Mam | 457 |  |  |  | 72.4% | 47.5% | 48.1% | 22.3% |
| test_reverb | tzijonel | Kaqchikel | 504 |  |  |  | 26.6% | 31.7% | 31.9% | 19.2% |
| test_phone | coerll | K'iche' | 359 |  |  |  | 6.1% | 12.8% | 13.1% | 9.7% |
| test_phone | fleurs | English | 300 |  |  |  | 100.0% | 100.0% | 100.0% | 100.0% |
| test_phone | fleurs | Spanish | 300 |  |  |  | 99.7% | 100.0% | 100.0% | 99.3% |
| test_phone | google_es | Spanish | 280 |  |  |  | 100.0% | 99.6% | 99.6% | 92.1% |
| test_phone | librispeech | English | 300 |  |  |  | 100.0% | 100.0% | 100.0% | 99.7% |
| test_phone | tachiwin | Mam | 457 |  |  |  | 57.1% | 49.5% | 49.5% | 19.3% |
| test_phone | tzijonel | Kaqchikel | 504 |  |  |  | 46.8% | 43.5% | 43.7% | 27.4% |

(FLEURS Spanish is in-corpus for bloom+aug+fleurs, because FLEURS-train is used.)

### Two-level decision (final model)

Name a language if confidence >= 0.99; otherwise say "Mayan: try X, then Y" if P(Mayan) >= 0.98; otherwise "not sure".

| condition | corpus | language | named | named correct | 'Mayan, try two' | contains true | not sure | Mayan routed to Spanish | useful answer |
|---|---|---|---|---|---|---|---|---|---|
| clean | coerll | K'iche' | 5.8% | 16.7% | 61.7% | 52.0% | 32.5% | 0.0% | 33.0% |
| clean | fleurs | English | 100.0% | 100.0% | 0.0% | - | 0.0% | - | - |
| clean | fleurs | Spanish | 99.6% | 100.0% | 0.0% | - | 0.4% | - | - |
| clean | google_es | Spanish | 90.0% | 100.0% | 0.0% | - | 10.0% | - | - |
| clean | librispeech | English | 100.0% | 100.0% | 0.0% | - | 0.0% | - | - |
| clean | tachiwin | Mam | 23.2% | 59.6% | 44.6% | 61.1% | 32.2% | 0.1% | 41.1% |
| clean | tzijonel | Kaqchikel | 26.7% | 48.9% | 50.0% | 83.4% | 23.4% | 4.8% | 54.7% |
| test_noise | coerll | K'iche' | 7.5% | 51.9% | 66.3% | 55.9% | 26.2% | 0.0% | 40.9% |
| test_noise | fleurs | English | 100.0% | 100.0% | 0.0% | - | 0.0% | - | - |
| test_noise | fleurs | Spanish | 99.7% | 100.0% | 0.0% | - | 0.3% | - | - |
| test_noise | google_es | Spanish | 91.1% | 100.0% | 0.0% | - | 8.9% | - | - |
| test_noise | librispeech | English | 99.3% | 100.0% | 0.0% | - | 0.7% | - | - |
| test_noise | tachiwin | Mam | 24.3% | 58.6% | 50.5% | 68.0% | 25.2% | 0.0% | 48.6% |
| test_noise | tzijonel | Kaqchikel | 24.6% | 39.5% | 53.0% | 84.6% | 22.4% | 5.0% | 54.6% |
| test_reverb | coerll | K'iche' | 4.7% | 35.3% | 62.7% | 45.3% | 32.6% | 0.0% | 30.1% |
| test_reverb | fleurs | English | 99.7% | 100.0% | 0.0% | - | 0.3% | - | - |
| test_reverb | fleurs | Spanish | 98.3% | 100.0% | 0.0% | - | 1.7% | - | - |
| test_reverb | google_es | Spanish | 67.1% | 100.0% | 0.0% | - | 32.9% | - | - |
| test_reverb | librispeech | English | 99.3% | 100.0% | 0.0% | - | 0.7% | - | - |
| test_reverb | tachiwin | Mam | 22.3% | 61.8% | 67.4% | 73.7% | 10.3% | 0.0% | 63.5% |
| test_reverb | tzijonel | Kaqchikel | 19.2% | 34.0% | 53.6% | 78.5% | 27.2% | 4.6% | 48.6% |
| test_phone | coerll | K'iche' | 9.7% | 11.4% | 68.8% | 32.4% | 21.4% | 0.0% | 23.4% |
| test_phone | fleurs | English | 100.0% | 100.0% | 0.0% | - | 0.0% | - | - |
| test_phone | fleurs | Spanish | 99.3% | 100.0% | 0.0% | - | 0.7% | - | - |
| test_phone | google_es | Spanish | 92.1% | 100.0% | 0.0% | - | 7.9% | - | - |
| test_phone | librispeech | English | 99.7% | 100.0% | 0.0% | - | 0.3% | - | - |
| test_phone | tachiwin | Mam | 19.3% | 62.5% | 46.8% | 69.2% | 33.9% | 0.0% | 44.4% |
| test_phone | tzijonel | Kaqchikel | 27.4% | 55.8% | 52.8% | 83.1% | 19.8% | 5.4% | 59.1% |

Held-out Bloom books: {'named_language': 0.5602770885028949, 'said_mayan_try_two': 0.320409429280397, 'said_not_sure': 0.11931348221670803, 'named_language_correct': 0.9647536445838716, 'try_two_family_correct': 0.9958050984188448, 'try_two_contains_true': 0.7757341077767022, 'mayan_routed_to_spanish': 0.005801906340654786, 'useful_answer': 0.7672330432380163}

## E5 Adding real-world recordings from each new corpus

Each new corpus is split in half by recording; k recordings per corpus from one half are added to training, the other half is the test (2403 clips). Mean of 3 random draws for k = 1-10.

| recordings added per corpus | clips added | coerll accuracy | tachiwin accuracy | tzijonel accuracy | fleurs accuracy | google_es accuracy | librispeech accuracy | coerll useful | tachiwin useful | tzijonel useful |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 19.8% | 41.4% | 42.2% | 100.0% | 100.0% | 100.0% | 27.0% | 43.0% | 55.7% |
| 1 | 60 | 44.2% | 73.5% | 51.5% | 100.0% | 100.0% | 100.0% | 49.4% | 78.4% | 63.9% |
| 2 | 124 | 54.5% | 70.7% | 58.6% | 100.0% | 100.0% | 100.0% | 55.6% | 76.1% | 68.9% |
| 5 | 317 | 69.7% | 83.1% | 71.4% | 100.0% | 100.0% | 100.0% | 71.9% | 87.5% | 77.9% |
| 10 | 594 | 84.8% | 87.6% | 80.5% | 100.0% | 100.0% | 100.0% | 79.7% | 92.0% | 85.3% |
| all | 2242 | 83.3% | 91.6% | 91.8% | 100.0% | 100.0% | 100.0% | 79.5% | 95.1% | 91.1% |

## E3 Languages it should decline

| corpus | language | n | declined | predicted when it answered |
|---|---|---|---|---|
| bloom | Tz'utujil | 41 | 100.0% | {} |
| chuj | Chuj | 59 | 72.9% | {'kjb': 9, 'mam': 6, 'spa': 1} |
| omnilingual | Itza' | 183 | 78.1% | {'kjb': 21, 'quc': 11, 'cak': 6, 'mam': 2} |
| omnilingual | Poqomam | 264 | 77.3% | {'cak': 33, 'quc': 20, 'spa': 5, 'kek': 1, 'mam': 1} |
| omnilingual | Sipakapense | 200 | 72.5% | {'quc': 51, 'cak': 3, 'mam': 1} |
| omnilingual | Sakapulteko | 208 | 82.2% | {'cak': 26, 'quc': 8, 'mam': 2, 'spa': 1} |
| tachiwin | Ch'ol | 149 | 81.2% | {'cak': 22, 'kjb': 3, 'mam': 2, 'quc': 1} |
| tachiwin | Huastec | 150 | 70.7% | {'mam': 34, 'quc': 4, 'spa': 4, 'kjb': 2} |
| tachiwin | Popti' (Jakalteko) | 141 | 61.0% | {'kjb': 53, 'cak': 1, 'mam': 1} |
| tachiwin | Tojolabal | 146 | 83.6% | {'cak': 15, 'kjb': 8, 'mam': 1} |
| tachiwin | Tseltal | 141 | 78.0% | {'kjb': 29, 'mam': 1, 'cak': 1} |
| tachiwin | Tsotsil | 148 | 86.5% | {'cak': 13, 'kjb': 5, 'mam': 1, 'spa': 1} |
| tachiwin | Yucatec Maya | 148 | 93.2% | {'cak': 4, 'spa': 4, 'mam': 1, 'eng': 1} |

English, held-out books (English is a class): accuracy 96.4%, sent to a Spanish interpreter 0.7%.

AUROC of confidence, target vs other languages: 0.727

## E4 Corpus shortcut probe

AUROC telling Bloom Spanish from FLEURS Spanish with the same features: 1.000 (near 1.0 means recording conditions are easy to see, so cross-corpus tests are essential).
