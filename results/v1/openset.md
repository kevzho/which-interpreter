# Open set: saying "I don't know"

Language-disjoint protocol: OTHER trained on 131 languages; method and threshold chosen on 26 tune languages; reported on 37 test languages never used before (ast, boz, bzi, Chuj, ces, dih, French, glg, guj, Popti' (Jakalteko), mal, mfy, mle, myk, ntp, ote, otm, ott, pbs, pmq, Poqomam, pol, Portuguese, Sipakapense, Sakapulteko, snk, tac, tar, toc, top, tos, tuz, Tz'utujil, ukr, vie, yaq, zul).

| score | tune AUROC | test AUROC (all 37) | test: 6 Guatemalan Mayan | test: 31 other languages |
|---|---|---|---|---|
| maxprob | 0.755 | 0.884 | 0.875 | 0.890 |
| oe (shipped) | 0.994 | 0.924 | 0.827 | 0.986 |
| mahalanobis | 0.991 | 0.889 | 0.755 | 0.973 |
| knn | 0.933 | 0.872 | 0.732 | 0.959 |
| oe+maha (best on tune; not deployable: rank fusion needs a reference set) | 0.996 | 0.928 | 0.833 | 0.987 |
| oe+knn | 0.988 | 0.922 | 0.822 | 0.984 |

Declined at each score's own tune threshold (90% of tune known-language clips answered):

| test language | maxprob | oe | mahalanobis | knn |
|---|---|---|---|---|
| Chuj | 13.6% | 94.9% | 33.9% | 35.6% |
| Popti' (Jakalteko) | 3.3% | 94.0% | 12.7% | 10.7% |
| Poqomam | 16.3% | 42.4% | 17.0% | 13.6% |
| Sakapulteko | 19.2% | 29.3% | 7.2% | 9.6% |
| Sipakapense | 24.0% | 27.5% | 24.0% | 18.0% |
| Tz'utujil | 19.5% | 22.0% | 58.5% | 9.8% |
| French | 11.4% | 100.0% | 100.0% | 92.1% |
| Portuguese | 22.8% | 98.9% | 100.0% | 32.6% |
| ast | 1.2% | 87.5% | 2.5% | 0.0% |
| boz | 42.9% | 100.0% | 100.0% | 92.9% |
| bzi | 43.3% | 100.0% | 100.0% | 100.0% |
| ces | 75.0% | 100.0% | 100.0% | 100.0% |
| dih | 20.7% | 100.0% | 100.0% | 96.6% |
| glg | 0.0% | 100.0% | 91.2% | 0.0% |
| guj | 4.2% | 100.0% | 100.0% | 58.3% |
| mal | 35.0% | 100.0% | 100.0% | 80.0% |
| mfy | 20.0% | 100.0% | 96.7% | 80.0% |
| mle | 0.0% | 100.0% | 100.0% | 83.3% |
| myk | 13.3% | 93.3% | 96.7% | 93.3% |
| ntp | 26.7% | 100.0% | 100.0% | 100.0% |
| ote | 26.7% | 100.0% | 90.0% | 90.0% |
| otm | 20.0% | 100.0% | 100.0% | 93.3% |
| ott | 23.3% | 100.0% | 100.0% | 90.0% |
| pbs | 34.8% | 100.0% | 95.7% | 87.0% |
| pmq | 43.3% | 100.0% | 100.0% | 96.7% |
| pol | 72.5% | 100.0% | 100.0% | 100.0% |
| snk | 28.3% | 95.0% | 100.0% | 93.3% |
| tac | 54.5% | 100.0% | 90.9% | 100.0% |
| tar | 20.0% | 100.0% | 96.7% | 93.3% |
| toc | 50.0% | 100.0% | 100.0% | 100.0% |
| top | 11.1% | 100.0% | 100.0% | 100.0% |
| tos | 10.0% | 100.0% | 100.0% | 93.3% |
| tuz | 38.9% | 97.2% | 100.0% | 97.2% |
| ukr | 3.8% | 100.0% | 100.0% | 100.0% |
| vie | 15.0% | 98.8% | 100.0% | 100.0% |
| yaq | 46.7% | 100.0% | 93.3% | 86.7% |
| zul | 11.2% | 100.0% | 100.0% | 100.0% |
| known languages answered (test) | 96.1% | 88.2% | 89.4% | 95.2% |

Shipped score: **oe**, threshold set to answer 90% of calibration known-language clips: held-out Bloom books plus one third of each new corpus's adaptation recordings, held out of the open-set model (so calibration includes field recordings, not only storybooks). The demo ships exactly this evaluated model and threshold (no refit), so tune and test languages remain unseen by the shipped model.
On test known-language clips it answers 88.2%, with 94.5% accuracy on those it answers. Accuracy without an OTHER class 91.2%; with it 90.0%.

Known-language test clips answered (not declined), by language:

| language | clips | answered | correct when answered |
|---|---|---|---|
| Spanish | 664 | 99.8% | 100.0% |
| Mam | 370 | 88.9% | 97.9% |
| K'iche' | 263 | 86.3% | 60.4% |
| Kaqchikel | 609 | 72.6% | 96.6% |
| English | 459 | 92.8% | 99.5% |

| test language | Mayan | clips | declined | of declined, flagged as Mayan |
|---|---|---|---|---|
| Chuj | yes | 59 | 94.9% | 82.1% |
| Popti' (Jakalteko) | yes | 150 | 94.0% | 95.7% |
| Poqomam | yes | 264 | 42.4% | 95.5% |
| Sakapulteko | yes | 208 | 29.3% | 93.4% |
| Sipakapense | yes | 200 | 27.5% | 98.2% |
| Tz'utujil | yes | 41 | 22.0% | 100.0% |
| French |  | 140 | 100.0% | 11.4% |
| Portuguese |  | 92 | 98.9% | 0.0% |
| ast |  | 80 | 87.5% | 0.0% |
| boz |  | 28 | 100.0% | 78.6% |
| bzi |  | 60 | 100.0% | 98.3% |
| ces |  | 80 | 100.0% | 17.5% |
| dih |  | 29 | 100.0% | 79.3% |
| glg |  | 80 | 100.0% | 0.0% |
| guj |  | 24 | 100.0% | 12.5% |
| mal |  | 60 | 100.0% | 36.7% |
| mfy |  | 30 | 100.0% | 63.3% |
| mle |  | 6 | 100.0% | 83.3% |
| myk |  | 60 | 93.3% | 91.1% |
| ntp |  | 30 | 100.0% | 80.0% |
| ote |  | 30 | 100.0% | 70.0% |
| otm |  | 30 | 100.0% | 40.0% |
| ott |  | 30 | 100.0% | 46.7% |
| pbs |  | 23 | 100.0% | 82.6% |
| pmq |  | 30 | 100.0% | 93.3% |
| pol |  | 80 | 100.0% | 56.2% |
| snk |  | 60 | 95.0% | 75.4% |
| tac |  | 11 | 100.0% | 90.9% |
| tar |  | 30 | 100.0% | 93.3% |
| toc |  | 2 | 100.0% | 100.0% |
| top |  | 9 | 100.0% | 100.0% |
| tos |  | 30 | 100.0% | 93.3% |
| tuz |  | 36 | 97.2% | 60.0% |
| ukr |  | 80 | 100.0% | 1.2% |
| vie |  | 80 | 98.8% | 0.0% |
| yaq |  | 30 | 100.0% | 86.7% |
| zul |  | 80 | 100.0% | 76.2% |
