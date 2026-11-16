# Validation of the shipped demo

Models scored: router.joblib `8a886a4f6bbb9839`, openset.joblib `ba5f7775a081993c` (SHA-256, frozen at 2026-11-15 11:25 EST before external scoring).

## 1. Is the open-set improvement more than noise?

Cluster bootstrap (2000 reps): 37 unknown test languages, 287 known clusters (recordings/books; FLEURS = one cluster per language).

### All test languages

| score | AUROC | 95% CI | diff vs max-prob | 95% CI of diff | P(not better) |
|---|---|---|---|---|---|
| maxprob | 0.895 | [0.800, 0.952] | - | - | - |
| oe | 0.927 | [0.871, 0.976] | +0.032 | [-0.032, 0.129] | 0.207 |
| mahalanobis | 0.900 | [0.807, 0.964] | +0.004 | [-0.060, 0.083] | 0.484 |
| knn | 0.879 | [0.772, 0.954] | -0.017 | [-0.084, 0.055] | 0.718 |
| oe+maha | 0.935 | [0.872, 0.978] | +0.039 | [-0.014, 0.123] | 0.089 |

### 6 Guatemalan Mayan (close relatives)

| score | AUROC | 95% CI | diff vs max-prob | 95% CI of diff | P(not better) |
|---|---|---|---|---|---|
| maxprob | 0.887 | [0.782, 0.947] | - | - | - |
| oe | 0.833 | [0.753, 0.909] | -0.054 | [-0.140, 0.067] | 0.853 |
| mahalanobis | 0.780 | [0.646, 0.874] | -0.108 | [-0.199, -0.049] | 1.000 |
| knn | 0.743 | [0.566, 0.858] | -0.144 | [-0.248, -0.078] | 1.000 |
| oe+maha | 0.850 | [0.759, 0.916] | -0.037 | [-0.109, 0.047] | 0.857 |

### Other test languages

| score | AUROC | 95% CI | diff vs max-prob | 95% CI of diff | P(not better) |
|---|---|---|---|---|---|
| maxprob | 0.901 | [0.807, 0.957] | - | - | - |
| oe | 0.986 | [0.970, 0.994] | +0.085 | [0.034, 0.172] | 0.000 |
| mahalanobis | 0.975 | [0.926, 0.995] | +0.074 | [0.026, 0.158] | 0.000 |
| knn | 0.963 | [0.913, 0.989] | +0.063 | [0.024, 0.130] | 0.000 |
| oe+maha | 0.988 | [0.964, 0.998] | +0.087 | [0.037, 0.175] | 0.000 |

## 2. Shipped pipeline end-to-end on held-out internal data

Held-out halves of the new corpora (same series as the adaptation half, likely same narrators), held-out English books, and open-set TEST languages (never trained on by the shipped models). 'Useful' = named correctly or 'Mayan, try two' containing the language (supported languages); declined (others). 95% CIs resample recordings/speakers.

| source | language | clusters | useful, 1 clip | useful, 2 clips | useful, 3 clips |
|---|---|---|---|---|---|
| bloom | English | 164 | 86.9% (82-92%) | 87.5% (81-94%) | 85.2% (74-93%) |
| coerll | K'iche' | 9 | 81.7% (78-91%) | 90.6% (86-100%) | 91.8% (89-100%) |
| fleurs | English | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| fleurs | Spanish | 1 | 99.8% (-) | 100.0% (-) | 100.0% (-) |
| google_es | Spanish | 18 | 78.5% (69-88%) | 87.5% (79-96%) | 91.7% (83-100%) |
| librispeech | English | 20 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| tachiwin | Mam | 30 | 88.6% (83-93%) | 92.7% (87-98%) | 91.2% (82-98%) |
| tzijonel | Kaqchikel | 44 | 70.9% (66-76%) | 74.4% (68-80%) | 78.5% (71-85%) |
| bloom | French | 16 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| bloom | Portuguese | 2 | 91.7% (83-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| bloom | Tz'utujil | 2 | 22.0% (19-25%) | 25.0% (20-30%) | 23.1% (14-33%) |
| bloom | boz | 7 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| bloom | bzi | 40 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| bloom | guj | 4 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| bloom | mal | 11 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| bloom | mle | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| bloom | myk | 34 | 93.3% (85-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| bloom | snk | 31 | 95.0% (89-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| bloom | tuz | 6 | 97.2% (92-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| chuj | Chuj | 2 | 94.9% (92-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| fleurs | French | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| fleurs | Portuguese | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| fleurs | ast | 1 | 82.5% (-) | 90.0% (-) | 92.3% (-) |
| fleurs | ces | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| fleurs | glg | 1 | 97.5% (-) | 100.0% (-) | 100.0% (-) |
| fleurs | pol | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| fleurs | ukr | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| fleurs | vie | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| fleurs | zul | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| omnilingual | Poqomam | 4 | 43.9% (26-64%) | 48.9% (27-72%) | 50.0% (24-81%) |
| omnilingual | Sakapulteko | 4 | 33.2% (23-45%) | 33.0% (22-51%) | 34.3% (21-61%) |
| omnilingual | Sipakapense | 4 | 28.5% (23-34%) | 22.2% (16-28%) | 27.7% (22-37%) |
| tachiwin | Popti' (Jakalteko) | 58 | 94.0% (90-97%) | 98.2% (95-100%) | 100.0% (100-100%) |
| tachiwin | dih | 12 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| tachiwin | mfy | 27 | 100.0% (100-100%) | 100.0% (100-100%) | - |
| tachiwin | ntp | 27 | 100.0% (100-100%) | 100.0% (100-100%) | - |
| tachiwin | ote | 26 | 100.0% (100-100%) | 100.0% (100-100%) | - |
| tachiwin | otm | 12 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| tachiwin | ott | 17 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| tachiwin | pbs | 9 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| tachiwin | pmq | 25 | 100.0% (100-100%) | 100.0% (100-100%) | - |
| tachiwin | tac | 5 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| tachiwin | tar | 24 | 100.0% (100-100%) | 100.0% (100-100%) | - |
| tachiwin | toc | 1 | 100.0% (-) | 100.0% (-) | - |
| tachiwin | top | 3 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| tachiwin | tos | 26 | 100.0% (100-100%) | 100.0% (100-100%) | - |
| tachiwin | yaq | 27 | 100.0% (100-100%) | 100.0% (100-100%) | - |

## 3. Fresh blind test (v2): Spanish/English sources and accents never used for anything

Google crowdsourced Spanish from ARGENTINA (accent held out), Multilingual LibriSpeech Spanish (audiobooks), People's Speech English (archive.org). Scored once with the frozen models above.

| source | language | speakers/recordings | useful (1 clip) | useful (3 clips) | named correctly (1 clip) | 'not supported' (1 clip) |
|---|---|---|---|---|---|---|
| google_es_ar | Spanish | 28 | 88.4% (83-93%) | 100.0% | 88.4% | 11.2% |
| mls | Spanish | 20 | 63.5% (51-76%) | 63.3% | 63.5% | 36.0% |
| peoples_speech | English | 9 | 83.3% (68-95%) | 100.0% | 83.3% | 16.7% |

Clips with no recognised phones (embedding-only): 13.

## 4. v1 external set (VoxPopuli, CIEMPIESS, EdAcc, Lingua Libre, Wikitongues)

This set was scored blind ONCE with the v1 models (results/v1/validation.md: e.g. only 9-13% of its Spanish was answered, because the open-set gate rejected unseen Spanish sources). That failure motivated v2, so the numbers below are a SECOND LOOK, not a blind test. v2 used no clip from it.

Single clips (first 8 s; clips under 1 s excluded because the demo refuses them: 10). The MMS-LID backbone itself was trained on FLEURS and evaluated on VoxPopuli by Meta, so 'never used' refers to this project's fitting and tuning.

| source | language | status | clusters | useful (1 clip) | useful (3 clips, same speaker) | named instead |
|---|---|---|---|---|---|---|
| ciempiess | Spanish | supported language | 20 | 18.0% (11-27%) | 10.0% |  |
| edacc | English | supported language | 20 | 47.9% (32-64%) | 53.3% |  |
| lingualibre | Kaqchikel | supported language | 1 | 12.5% (-) | 0.0% |  |
| voxpopuli | English | supported language | 121 | 38.7% (29-47%) | 25.0% |  |
| voxpopuli | Spanish | supported language | 78 | 10.0% (5-15%) | 0.0% |  |
| wikitongues | K'iche' | supported language | 1 | 25.0% (-) | 100.0% |  |
| wikitongues | Spanish | supported language | 1 | 50.0% (-) | 100.0% |  |
| voxpopuli | French | never seen | 92 | 100.0% (100-100%) | 100.0% | {} |
| voxpopuli | est | never seen | 6 | 100.0% (100-100%) | 100.0% | {} |
| voxpopuli | hrv | never seen | 21 | 100.0% (100-100%) | 100.0% | {} |
| voxpopuli | lit | never seen | 7 | 100.0% (100-100%) | 100.0% | {} |
| voxpopuli | pol | never seen | 77 | 100.0% (100-100%) | 100.0% | {} |
| voxpopuli | slk | never seen | 26 | 100.0% (100-100%) | 100.0% | {} |
| wikitongues | Garifuna | never seen | 1 | 100.0% (-) | 100.0% | {} |
| wikitongues | Lacandon | never seen | 1 | 100.0% (-) | 100.0% | {} |
| wikitongues | Sakapulteko | never seen | 1 | 75.0% (-) | 100.0% | {"K'iche'": 1} |
| voxpopuli | ita | tuning only | 76 | 100.0% (100-100%) | 100.0% | {} |
| voxpopuli | deu | trained as OTHER | 92 | 100.0% (100-100%) | 100.0% | {} |
| voxpopuli | fin | trained as OTHER | 27 | 100.0% (100-100%) | 100.0% | {} |
| voxpopuli | ron | trained as OTHER | 48 | 100.0% (100-100%) | 100.0% | {} |
| wikitongues | Yucatec Maya | trained as OTHER | 2 | 81.8% (75-100%) | 100.0% | {'Kaqchikel': 2} |

External AUROC (known vs unknown, single clips; 242 known speakers):
- all unknown languages, maxprob: 0.904 [0.856, 0.942]
- all unknown languages, oe: 0.984 [0.967, 0.992]; diff vs max-prob +0.080 [0.041, 0.126]
- never-seen unknown languages only, maxprob: 0.926 [0.882, 0.957]
- never-seen unknown languages only, oe: 0.980 [0.947, 0.991]; diff vs max-prob +0.054 [0.014, 0.092]
- VoxPopuli only (same source both sides), maxprob: 0.902 [0.843, 0.946]
- VoxPopuli only (same source both sides), oe: 0.987 [0.979, 0.994]; diff vs max-prob +0.086 [0.041, 0.142]
- VoxPopuli only, never-seen languages, maxprob: 0.929 [0.881, 0.965]
- VoxPopuli only, never-seen languages, oe: 0.984 [0.974, 0.991]; diff vs max-prob +0.054 [0.020, 0.100]
