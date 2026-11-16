# Validation of the shipped demo

Models scored: router.joblib `78f0918ea84e0b7e`, openset.joblib `f54c3a4aa7394426` (SHA-256, frozen at 2026-11-15 08:47 EST before external scoring).

## 1. Is the open-set improvement more than noise?

Cluster bootstrap (2000 reps): 37 unknown test languages, 249 known clusters (recordings/books; FLEURS = one cluster per language).

### All test languages

| score | AUROC | 95% CI | diff vs max-prob | 95% CI of diff | P(not better) |
|---|---|---|---|---|---|
| maxprob | 0.884 | [0.774, 0.951] | - | - | - |
| oe | 0.924 | [0.863, 0.977] | +0.040 | [-0.029, 0.145] | 0.165 |
| mahalanobis | 0.889 | [0.779, 0.963] | +0.004 | [-0.068, 0.090] | 0.484 |
| knn | 0.872 | [0.753, 0.953] | -0.013 | [-0.081, 0.063] | 0.666 |
| oe+maha | 0.928 | [0.856, 0.978] | +0.044 | [-0.016, 0.136] | 0.098 |

### 6 Guatemalan Mayan (close relatives)

| score | AUROC | 95% CI | diff vs max-prob | 95% CI of diff | P(not better) |
|---|---|---|---|---|---|
| maxprob | 0.875 | [0.752, 0.942] | - | - | - |
| oe | 0.827 | [0.740, 0.907] | -0.048 | [-0.135, 0.083] | 0.811 |
| mahalanobis | 0.755 | [0.595, 0.867] | -0.121 | [-0.233, -0.048] | 1.000 |
| knn | 0.732 | [0.541, 0.859] | -0.143 | [-0.250, -0.073] | 1.000 |
| oe+maha | 0.834 | [0.727, 0.911] | -0.041 | [-0.125, 0.049] | 0.842 |

### Other test languages

| score | AUROC | 95% CI | diff vs max-prob | 95% CI of diff | P(not better) |
|---|---|---|---|---|---|
| maxprob | 0.890 | [0.770, 0.954] | - | - | - |
| oe | 0.986 | [0.970, 0.994] | +0.096 | [0.037, 0.211] | 0.000 |
| mahalanobis | 0.973 | [0.916, 0.995] | +0.082 | [0.027, 0.185] | 0.000 |
| knn | 0.959 | [0.901, 0.988] | +0.069 | [0.026, 0.158] | 0.000 |
| oe+maha | 0.987 | [0.962, 0.998] | +0.097 | [0.039, 0.211] | 0.000 |

## 2. Shipped pipeline end-to-end on held-out internal data

Held-out halves of the new corpora (same series as the adaptation half, likely same narrators), held-out English books, and open-set TEST languages (never trained on by the shipped models). 'Useful' = named correctly or 'Mayan, try two' containing the language (supported languages); declined (others). 95% CIs resample recordings/speakers.

| source | language | clusters | useful, 1 clip | useful, 2 clips | useful, 3 clips |
|---|---|---|---|---|---|
| bloom | English | 164 | 86.6% (81-92%) | 87.5% (81-94%) | 85.2% (76-94%) |
| coerll | K'iche' | 9 | 82.1% (78-91%) | 90.6% (87-100%) | 91.8% (89-100%) |
| fleurs | English | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| fleurs | Spanish | 1 | 99.8% (-) | 100.0% (-) | 100.0% (-) |
| tachiwin | Mam | 30 | 88.6% (83-93%) | 93.3% (87-98%) | 95.6% (88-100%) |
| tzijonel | Kaqchikel | 44 | 71.3% (66-76%) | 74.4% (68-80%) | 77.9% (70-85%) |
| bloom | French | 16 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| bloom | Portuguese | 2 | 91.7% (83-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| bloom | Tz'utujil | 2 | 22.0% (19-25%) | 25.0% (20-30%) | 15.4% (14-17%) |
| bloom | boz | 7 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| bloom | bzi | 40 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| bloom | guj | 4 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| bloom | mal | 11 | 100.0% (100-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| bloom | mle | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| bloom | myk | 34 | 95.0% (88-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| bloom | snk | 31 | 95.0% (89-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| bloom | tuz | 6 | 97.2% (92-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| chuj | Chuj | 2 | 94.9% (92-100%) | 100.0% (100-100%) | 100.0% (100-100%) |
| fleurs | French | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| fleurs | Portuguese | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| fleurs | ast | 1 | 87.5% (-) | 95.0% (-) | 96.2% (-) |
| fleurs | ces | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| fleurs | glg | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| fleurs | pol | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| fleurs | ukr | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| fleurs | vie | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| fleurs | zul | 1 | 100.0% (-) | 100.0% (-) | 100.0% (-) |
| omnilingual | Poqomam | 4 | 43.2% (25-63%) | 45.8% (22-71%) | 52.3% (24-83%) |
| omnilingual | Sakapulteko | 4 | 29.3% (20-44%) | 31.1% (17-57%) | 32.8% (19-58%) |
| omnilingual | Sipakapense | 4 | 27.5% (21-34%) | 29.3% (22-36%) | 26.2% (12-39%) |
| tachiwin | Popti' (Jakalteko) | 58 | 94.0% (90-98%) | 98.2% (95-100%) | 100.0% (100-100%) |
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

Clips with no recognised phones (embedding-only): 13.

## 3. Frozen external validation (sources never used for fitting or tuning in this project)

Single clips (first 8 s; clips under 1 s excluded because the demo refuses them: 10). The MMS-LID backbone itself was trained on FLEURS and evaluated on VoxPopuli by Meta, so 'never used' refers to this project's fitting and tuning.

| source | language | status | clusters | useful (1 clip) | useful (3 clips, same speaker) | named instead |
|---|---|---|---|---|---|---|
| ciempiess | Spanish | supported language | 20 | 13.0% (7-20%) | 4.0% |  |
| edacc | English | supported language | 20 | 47.9% (31-63%) | 53.3% |  |
| lingualibre | Kaqchikel | supported language | 1 | 12.5% (-) | 0.0% |  |
| voxpopuli | English | supported language | 121 | 38.7% (30-48%) | 25.0% |  |
| voxpopuli | Spanish | supported language | 78 | 8.7% (4-14%) | 0.0% |  |
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
- all unknown languages, maxprob: 0.899 [0.851, 0.938]
- all unknown languages, oe: 0.983 [0.965, 0.992]; diff vs max-prob +0.083 [0.042, 0.129]
- never-seen unknown languages only, maxprob: 0.923 [0.876, 0.955]
- never-seen unknown languages only, oe: 0.979 [0.942, 0.990]; diff vs max-prob +0.056 [0.016, 0.095]
- VoxPopuli only (same source both sides), maxprob: 0.896 [0.841, 0.942]
- VoxPopuli only (same source both sides), oe: 0.986 [0.977, 0.994]; diff vs max-prob +0.090 [0.044, 0.144]
- VoxPopuli only, never-seen languages, maxprob: 0.925 [0.876, 0.963]
- VoxPopuli only, never-seen languages, oe: 0.983 [0.972, 0.991]; diff vs max-prob +0.057 [0.023, 0.104]
