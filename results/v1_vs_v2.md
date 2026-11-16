# v1 vs v2 models on the same clips

Both versions frozen; single clips; clips under 1 s excluded. 'Router right when rejected' = among clips the 'I don't know' gate rejected, how often the 7-class language router's top guess was the correct language.

| clips | n | v1 useful | v2 useful | v1 rejected | v2 rejected | router right when rejected (v1 / v2) |
|---|---|---|---|---|---|---|
| Spanish, Argentina (blind v2) | 224 | 87.9% | 88.4% | 12.1% | 11.2% | 100.0% / 100.0% |
| Spanish, audiobooks (blind v2) | 200 | 58.0% | 63.5% | 41.0% | 36.0% | 100.0% / 100.0% |
| English, archive.org (blind v2) | 42 | 83.3% | 83.3% | 16.7% | 16.7% | 100.0% / 100.0% |
| Spanish, radio (v1 external) | 300 | 13.0% | 18.0% | 87.0% | 82.0% | 98.5% / 98.8% |
| Spanish, parliament (v1 external) | 150 | 8.7% | 10.0% | 90.7% | 89.3% | 93.4% / 94.0% |
| English, parliament (v1 external) | 150 | 38.7% | 38.7% | 61.3% | 61.3% | 97.8% / 97.8% |
| English, conversation (v1 external) | 169 | 47.9% | 47.9% | 49.1% | 49.1% | 77.1% / 77.1% |
