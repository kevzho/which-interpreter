# Shipped model (embedding + phones, adapted)

Fusion weight on the embedding model: 0.5 (chosen on held-out Bloom books). Thresholds: language 0.85, branch 0.82, Mayan 0.83.
Held-out Bloom books macro-F1: embedding 0.788 -> fused 0.802.

## Held-out recordings of the new corpora

| corpus | n | accuracy | named a language | named correctly | 'Mayan, try two' | contains true | not sure | Mayan sent to Spanish | useful answer |
|---|---|---|---|---|---|---|---|---|---|
| coerll | 263 | 85.6% | 54.4% | 100.0% | 41.1% | 93.5% | 4.6% | 0.0% | 92.8% |
| tachiwin | 370 | 92.7% | 90.0% | 98.5% | 9.5% | 82.9% | 0.5% | 0.5% | 96.5% |
| tzijonel | 609 | 92.8% | 82.4% | 97.2% | 13.0% | 98.7% | 4.6% | 2.0% | 92.9% |
| fleurs | 817 | 100.0% | 100.0% | 100.0% | 0.0% | - | 0.0% | - | - |
| google_es | 144 | 99.3% | 98.6% | 100.0% | 0.0% | - | 1.4% | - | - |
| librispeech | 200 | 100.0% | 100.0% | 100.0% | 0.0% | - | 0.0% | - | - |

English, held-out books: 97.1% correct, 0.7% sent to Spanish.

## Languages it has no class for (router alone, before the "not supported" check; full app: validation.md)

| language | nearest covered relative | outcomes |
|---|---|---|
| Tz'utujil | Kaqchikel, K'iche' | {'nearest branch': 0.61, 'said Mayan': 0.22, 'named nearest relative': 0.146, 'declined': 0.024} |
| Itza' | none close | {'said Mayan': 0.574, 'named unrelated language': 0.284, 'other branch': 0.098, 'declined': 0.044} |
| Poqomam | Kaqchikel, Q'eqchi', K'iche' | {'nearest branch': 0.333, 'named nearest relative': 0.311, 'said Mayan': 0.258, 'declined': 0.064, 'said Spanish/English': 0.03, 'named unrelated language': 0.004} |
| Sipakapense | Kaqchikel, K'iche' | {'named nearest relative': 0.345, 'nearest branch': 0.33, 'said Mayan': 0.3, 'named unrelated language': 0.015, 'declined': 0.005, 'other branch': 0.005} |
| Sakapulteko | Kaqchikel, K'iche' | {'nearest branch': 0.553, 'named nearest relative': 0.226, 'said Mayan': 0.183, 'said Spanish/English': 0.019, 'declined': 0.01, 'named unrelated language': 0.01} |
| Popti' (Jakalteko) | Q'anjob'al | {'named nearest relative': 0.66, 'said Mayan': 0.234, 'nearest branch': 0.05, 'named unrelated language': 0.028, 'declined': 0.028} |
| Yucatec Maya | none close | {'said Mayan': 0.595, 'declined': 0.176, 'other branch': 0.108, 'named unrelated language': 0.068, 'said Spanish/English': 0.054} |
| Tseltal | none close | {'said Mayan': 0.383, 'named unrelated language': 0.362, 'declined': 0.227, 'other branch': 0.021, 'said Spanish/English': 0.007} |
| Tsotsil | none close | {'said Mayan': 0.541, 'named unrelated language': 0.216, 'declined': 0.115, 'other branch': 0.108, 'said Spanish/English': 0.02} |
| Ch'ol | none close | {'said Mayan': 0.463, 'named unrelated language': 0.295, 'declined': 0.154, 'other branch': 0.087} |
| Tojolabal | Q'anjob'al | {'said Mayan': 0.603, 'named nearest relative': 0.192, 'named unrelated language': 0.116, 'declined': 0.041, 'other branch': 0.041, 'nearest branch': 0.007} |
| Huastec | none close | {'said Mayan': 0.467, 'named unrelated language': 0.387, 'other branch': 0.06, 'declined': 0.047, 'said Spanish/English': 0.04} |
| Chuj | Q'anjob'al | {'said Mayan': 0.458, 'said Spanish/English': 0.186, 'named unrelated language': 0.153, 'named nearest relative': 0.119, 'nearest branch': 0.034, 'other branch': 0.034, 'declined': 0.017} |
