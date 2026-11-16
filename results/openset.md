# Open set: saying "I don't know"

Language-disjoint protocol: OTHER trained on 131 languages; method and threshold chosen on 26 tune languages; reported on 37 test languages never used before (ast, boz, bzi, Chuj, ces, dih, French, glg, guj, Popti' (Jakalteko), mal, mfy, mle, myk, ntp, ote, otm, ott, pbs, pmq, Poqomam, pol, Portuguese, Sipakapense, Sakapulteko, snk, tac, tar, toc, top, tos, tuz, Tz'utujil, ukr, vie, yaq, zul).

| score | tune AUROC | test AUROC (all 37) | test: 6 Guatemalan Mayan | test: 31 other languages |
|---|---|---|---|---|
| maxprob | 0.759 | 0.895 | 0.887 | 0.901 |
| oe (shipped) | 0.994 | 0.927 | 0.833 | 0.986 |
| mahalanobis | 0.992 | 0.900 | 0.780 | 0.975 |
| knn | 0.936 | 0.879 | 0.743 | 0.963 |
| oe+maha (best on tune; not deployable: rank fusion needs a reference set) | 0.996 | 0.935 | 0.849 | 0.988 |
| oe+knn | 0.988 | 0.926 | 0.830 | 0.985 |

Declined at each score's own tune threshold (90% of tune known-language clips answered):

| test language | maxprob | oe | mahalanobis | knn |
|---|---|---|---|---|
| Chuj | 13.6% | 94.9% | 33.9% | 35.6% |
| Popti' (Jakalteko) | 3.3% | 94.0% | 13.3% | 10.7% |
| Poqomam | 18.2% | 43.2% | 17.8% | 13.6% |
| Sakapulteko | 18.8% | 33.2% | 7.7% | 9.1% |
| Sipakapense | 24.0% | 28.5% | 24.5% | 18.0% |
| Tz'utujil | 24.4% | 22.0% | 61.0% | 9.8% |
| French | 13.6% | 100.0% | 100.0% | 93.6% |
| Portuguese | 28.3% | 98.9% | 100.0% | 42.4% |
| ast | 0.0% | 82.5% | 2.5% | 0.0% |
| boz | 46.4% | 100.0% | 100.0% | 92.9% |
| bzi | 45.0% | 100.0% | 100.0% | 100.0% |
| ces | 76.2% | 100.0% | 100.0% | 100.0% |
| dih | 10.3% | 100.0% | 100.0% | 96.6% |
| glg | 0.0% | 97.5% | 91.2% | 0.0% |
| guj | 4.2% | 100.0% | 100.0% | 66.7% |
| mal | 31.7% | 100.0% | 100.0% | 80.0% |
| mfy | 13.3% | 100.0% | 96.7% | 83.3% |
| mle | 0.0% | 100.0% | 100.0% | 100.0% |
| myk | 13.3% | 91.7% | 96.7% | 93.3% |
| ntp | 30.0% | 100.0% | 100.0% | 100.0% |
| ote | 30.0% | 100.0% | 90.0% | 90.0% |
| otm | 20.0% | 100.0% | 100.0% | 96.7% |
| ott | 26.7% | 100.0% | 100.0% | 90.0% |
| pbs | 43.5% | 100.0% | 95.7% | 87.0% |
| pmq | 43.3% | 100.0% | 100.0% | 96.7% |
| pol | 72.5% | 100.0% | 100.0% | 100.0% |
| snk | 30.0% | 95.0% | 100.0% | 93.3% |
| tac | 54.5% | 100.0% | 100.0% | 100.0% |
| tar | 26.7% | 100.0% | 96.7% | 93.3% |
| toc | 50.0% | 100.0% | 100.0% | 100.0% |
| top | 11.1% | 100.0% | 100.0% | 100.0% |
| tos | 10.0% | 100.0% | 100.0% | 93.3% |
| tuz | 38.9% | 97.2% | 100.0% | 97.2% |
| ukr | 3.8% | 100.0% | 100.0% | 100.0% |
| vie | 11.2% | 98.8% | 100.0% | 100.0% |
| yaq | 50.0% | 100.0% | 93.3% | 86.7% |
| zul | 12.5% | 100.0% | 100.0% | 100.0% |
| known languages answered (test) | 96.3% | 88.5% | 90.7% | 95.3% |

Shipped score: **oe**, threshold set to answer 90% of calibration known-language clips: held-out Bloom books plus one third of each new corpus's adaptation recordings, held out of the open-set model (so calibration includes field recordings, not only storybooks). The demo ships exactly this evaluated model and threshold (no refit), so tune and test languages remain unseen by the shipped model.
On test known-language clips it answers 88.5%, with 95.2% accuracy on those it answers. Accuracy without an OTHER class 92.4%; with it 91.1%.

Known-language test clips answered (not declined), by language:

| language | clips | answered | correct when answered |
|---|---|---|---|
| Spanish | 808 | 96.0% | 100.0% |
| Mam | 370 | 88.9% | 97.9% |
| K'iche' | 263 | 86.3% | 60.4% |
| Kaqchikel | 609 | 72.2% | 96.6% |
| English | 659 | 95.0% | 99.7% |

| test language | Mayan | clips | declined | of declined, flagged as Mayan |
|---|---|---|---|---|
| Chuj | yes | 59 | 94.9% | 82.1% |
| Popti' (Jakalteko) | yes | 150 | 94.0% | 97.2% |
| Poqomam | yes | 264 | 43.2% | 94.7% |
| Sakapulteko | yes | 208 | 33.2% | 94.2% |
| Sipakapense | yes | 200 | 28.5% | 98.2% |
| Tz'utujil | yes | 41 | 22.0% | 100.0% |
| French |  | 140 | 100.0% | 11.4% |
| Portuguese |  | 92 | 98.9% | 0.0% |
| ast |  | 80 | 82.5% | 0.0% |
| boz |  | 28 | 100.0% | 78.6% |
| bzi |  | 60 | 100.0% | 98.3% |
| ces |  | 80 | 100.0% | 13.8% |
| dih |  | 29 | 100.0% | 79.3% |
| glg |  | 80 | 97.5% | 0.0% |
| guj |  | 24 | 100.0% | 12.5% |
| mal |  | 60 | 100.0% | 36.7% |
| mfy |  | 30 | 100.0% | 63.3% |
| mle |  | 6 | 100.0% | 83.3% |
| myk |  | 60 | 91.7% | 90.9% |
| ntp |  | 30 | 100.0% | 83.3% |
| ote |  | 30 | 100.0% | 70.0% |
| otm |  | 30 | 100.0% | 43.3% |
| ott |  | 30 | 100.0% | 46.7% |
| pbs |  | 23 | 100.0% | 82.6% |
| pmq |  | 30 | 100.0% | 93.3% |
| pol |  | 80 | 100.0% | 57.5% |
| snk |  | 60 | 95.0% | 73.7% |
| tac |  | 11 | 100.0% | 90.9% |
| tar |  | 30 | 100.0% | 96.7% |
| toc |  | 2 | 100.0% | 100.0% |
| top |  | 9 | 100.0% | 100.0% |
| tos |  | 30 | 100.0% | 93.3% |
| tuz |  | 36 | 97.2% | 60.0% |
| ukr |  | 80 | 100.0% | 1.2% |
| vie |  | 80 | 98.8% | 0.0% |
| yaq |  | 30 | 100.0% | 86.7% |
| zul |  | 80 | 100.0% | 81.2% |
