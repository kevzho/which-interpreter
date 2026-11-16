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
| clean | tachiwin | Mam | 715 | 0.0% | 39.0% | 4.5% | 42.1% | 39.7% | 42.1% | 23.2% |
| clean | tzijonel | Kaqchikel | 1219 | 0.0% | 49.8% | 3.4% | 39.7% | 42.3% | 42.5% | 26.7% |
| test_noise | coerll | K'iche' | 299 |  |  |  | 30.4% | 30.4% | 32.4% | 7.7% |
| test_noise | fleurs | English | 300 |  |  |  | 100.0% | 100.0% | 100.0% | 100.0% |
| test_noise | fleurs | Spanish | 300 |  |  |  | 100.0% | 100.0% | 100.0% | 99.7% |
| test_noise | tachiwin | Mam | 268 |  |  |  | 54.5% | 41.8% | 42.5% | 22.8% |
| test_noise | tzijonel | Kaqchikel | 295 |  |  |  | 34.9% | 36.3% | 36.6% | 25.8% |
| test_reverb | coerll | K'iche' | 299 |  |  |  | 18.1% | 21.4% | 23.1% | 5.0% |
| test_reverb | fleurs | English | 300 |  |  |  | 100.0% | 100.0% | 100.0% | 99.7% |
| test_reverb | fleurs | Spanish | 300 |  |  |  | 99.7% | 100.0% | 100.0% | 98.3% |
| test_reverb | tachiwin | Mam | 268 |  |  |  | 74.3% | 48.5% | 50.4% | 22.0% |
| test_reverb | tzijonel | Kaqchikel | 295 |  |  |  | 28.5% | 31.9% | 32.2% | 19.3% |
| test_phone | coerll | K'iche' | 299 |  |  |  | 6.4% | 12.4% | 12.7% | 9.4% |
| test_phone | fleurs | English | 300 |  |  |  | 100.0% | 100.0% | 100.0% | 100.0% |
| test_phone | fleurs | Spanish | 300 |  |  |  | 99.7% | 100.0% | 100.0% | 99.3% |
| test_phone | tachiwin | Mam | 268 |  |  |  | 57.8% | 49.3% | 48.9% | 16.0% |
| test_phone | tzijonel | Kaqchikel | 295 |  |  |  | 47.1% | 42.0% | 42.4% | 27.5% |

(FLEURS Spanish is in-corpus for bloom+aug+fleurs, because FLEURS-train is used.)

### Two-level decision (final model)

Name a language if confidence >= 0.99; otherwise say "Mayan: try X, then Y" if P(Mayan) >= 0.98; otherwise "not sure".

| condition | corpus | language | named | named correct | 'Mayan, try two' | contains true | not sure | Mayan routed to Spanish | useful answer |
|---|---|---|---|---|---|---|---|---|---|
| clean | coerll | K'iche' | 5.8% | 16.7% | 61.7% | 52.0% | 32.5% | 0.0% | 33.0% |
| clean | fleurs | English | 100.0% | 100.0% | 0.0% | - | 0.0% | - | - |
| clean | fleurs | Spanish | 99.6% | 100.0% | 0.0% | - | 0.4% | - | - |
| clean | tachiwin | Mam | 23.2% | 59.6% | 44.6% | 61.1% | 32.2% | 0.1% | 41.1% |
| clean | tzijonel | Kaqchikel | 26.7% | 48.9% | 50.0% | 83.4% | 23.4% | 4.8% | 54.7% |
| test_noise | coerll | K'iche' | 7.7% | 43.5% | 66.9% | 56.0% | 25.4% | 0.0% | 40.8% |
| test_noise | fleurs | English | 100.0% | 100.0% | 0.0% | - | 0.0% | - | - |
| test_noise | fleurs | Spanish | 99.7% | 100.0% | 0.0% | - | 0.3% | - | - |
| test_noise | tachiwin | Mam | 22.8% | 57.4% | 51.9% | 70.5% | 25.4% | 0.0% | 49.6% |
| test_noise | tzijonel | Kaqchikel | 25.8% | 36.8% | 52.5% | 87.1% | 21.7% | 5.1% | 55.3% |
| test_reverb | coerll | K'iche' | 5.0% | 33.3% | 62.2% | 43.5% | 32.8% | 0.0% | 28.8% |
| test_reverb | fleurs | English | 99.7% | 100.0% | 0.0% | - | 0.3% | - | - |
| test_reverb | fleurs | Spanish | 98.3% | 100.0% | 0.0% | - | 1.7% | - | - |
| test_reverb | tachiwin | Mam | 22.0% | 62.7% | 68.7% | 74.5% | 9.3% | 0.0% | 64.9% |
| test_reverb | tzijonel | Kaqchikel | 19.3% | 35.1% | 52.9% | 76.9% | 27.8% | 3.7% | 47.5% |
| test_phone | coerll | K'iche' | 9.4% | 10.7% | 69.6% | 31.2% | 21.1% | 0.0% | 22.7% |
| test_phone | fleurs | English | 100.0% | 100.0% | 0.0% | - | 0.0% | - | - |
| test_phone | fleurs | Spanish | 99.3% | 100.0% | 0.0% | - | 0.7% | - | - |
| test_phone | tachiwin | Mam | 16.0% | 69.8% | 52.2% | 72.9% | 31.7% | 0.0% | 49.3% |
| test_phone | tzijonel | Kaqchikel | 27.5% | 58.0% | 53.2% | 82.2% | 19.3% | 4.7% | 59.7% |

Held-out Bloom books: {'named_language': 0.5602770885028949, 'said_mayan_try_two': 0.320409429280397, 'said_not_sure': 0.11931348221670803, 'named_language_correct': 0.9647536445838716, 'try_two_family_correct': 0.9958050984188448, 'try_two_contains_true': 0.7757341077767022, 'mayan_routed_to_spanish': 0.005801906340654786, 'useful_answer': 0.7672330432380163}

## E5 Adding real-world recordings from each new corpus

Each new corpus is split in half by recording; k recordings per corpus from one half are added to training, the other half is the test (2059 clips). Mean of 3 random draws for k = 1-10.

| recordings added per corpus | clips added | coerll accuracy | tachiwin accuracy | tzijonel accuracy | fleurs accuracy | coerll useful | tachiwin useful | tzijonel useful |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 19.8% | 41.4% | 42.2% | 100.0% | 27.0% | 43.0% | 55.7% |
| 1 | 44 | 47.4% | 65.3% | 53.3% | 100.0% | 51.5% | 69.8% | 64.1% |
| 2 | 99 | 59.8% | 74.8% | 62.9% | 100.0% | 60.7% | 78.1% | 70.4% |
| 5 | 225 | 76.4% | 83.7% | 73.7% | 100.0% | 74.8% | 87.6% | 79.7% |
| 10 | 427 | 85.0% | 86.3% | 79.7% | 100.0% | 79.7% | 89.8% | 84.7% |
| all | 1906 | 83.7% | 91.6% | 91.8% | 100.0% | 79.8% | 95.1% | 91.3% |

## E3 Languages it should decline

| corpus | language | n | declined | predicted when it answered |
|---|---|---|---|---|
| bloom | Tz'utujil | 41 | 100.0% | {} |
| chuj | Chuj | 59 | 72.9% | {'kjb': 8, 'mam': 7, 'spa': 1} |
| omnilingual | Itza' | 183 | 78.7% | {'kjb': 21, 'quc': 10, 'cak': 6, 'mam': 2} |
| omnilingual | Poqomam | 264 | 77.7% | {'cak': 33, 'quc': 20, 'spa': 5, 'mam': 1} |
| omnilingual | Sipakapense | 200 | 73.0% | {'quc': 51, 'cak': 2, 'mam': 1} |
| omnilingual | Sakapulteko | 208 | 82.7% | {'cak': 25, 'quc': 8, 'mam': 2, 'spa': 1} |
| tachiwin | Ch'ol | 149 | 81.2% | {'cak': 22, 'kjb': 3, 'mam': 2, 'quc': 1} |
| tachiwin | Huastec | 150 | 68.7% | {'mam': 37, 'quc': 4, 'spa': 4, 'kjb': 2} |
| tachiwin | Popti' (Jakalteko) | 141 | 62.4% | {'kjb': 50, 'cak': 1, 'mam': 1, 'spa': 1} |
| tachiwin | Tojolabal | 146 | 83.6% | {'cak': 15, 'kjb': 8, 'mam': 1} |
| tachiwin | Tseltal | 141 | 76.6% | {'kjb': 31, 'mam': 1, 'cak': 1} |
| tachiwin | Tsotsil | 148 | 86.5% | {'cak': 13, 'kjb': 5, 'mam': 1, 'spa': 1} |
| tachiwin | Yucatec Maya | 148 | 93.2% | {'cak': 4, 'spa': 4, 'mam': 1, 'eng': 1} |

English, held-out books (English is a class): accuracy 96.4%, sent to a Spanish interpreter 1.0%.

AUROC of confidence, target vs other languages: 0.721

## E4 Corpus shortcut probe

AUROC telling Bloom Spanish from FLEURS Spanish with the same features: 1.000 (near 1.0 means recording conditions are easy to see, so cross-corpus tests are essential).
