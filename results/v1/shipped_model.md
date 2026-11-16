# Shipped model (embedding + phones, adapted)

Fusion weight on the embedding model: 0.5 (chosen on held-out Bloom books). Thresholds: language 0.85, branch 0.82, Mayan 0.83.
Held-out Bloom books macro-F1: embedding 0.788 -> fused 0.802.

## Held-out recordings of the new corpora

| corpus | n | accuracy | named a language | named correctly | 'Mayan, try two' | contains true | not sure | Mayan sent to Spanish | useful answer |
|---|---|---|---|---|---|---|---|---|---|
| coerll | 263 | 85.6% | 52.9% | 100.0% | 42.6% | 95.5% | 4.6% | 0.0% | 93.5% |
| tachiwin | 370 | 93.2% | 89.7% | 98.5% | 9.7% | 83.3% | 0.5% | 0.5% | 96.5% |
| tzijonel | 609 | 93.1% | 82.6% | 97.0% | 13.0% | 98.7% | 4.4% | 2.1% | 92.9% |
| fleurs | 817 | 100.0% | 100.0% | 100.0% | 0.0% | - | 0.0% | - | - |

English, held-out books: 97.1% correct, 0.7% sent to Spanish.

## Languages it has no class for

| language | nearest covered relative | outcomes |
|---|---|---|
| Tz'utujil | Kaqchikel, K'iche' | {'nearest branch': 0.61, 'said Mayan': 0.22, 'named nearest relative': 0.146, 'declined': 0.024} |
| Itza' | none close | {'said Mayan': 0.579, 'named unrelated language': 0.29, 'other branch': 0.087, 'declined': 0.044} |
| Poqomam | Kaqchikel, Q'eqchi', K'iche' | {'named nearest relative': 0.322, 'nearest branch': 0.318, 'said Mayan': 0.261, 'declined': 0.064, 'said Spanish/English': 0.03, 'named unrelated language': 0.004} |
| Sipakapense | Kaqchikel, K'iche' | {'named nearest relative': 0.345, 'nearest branch': 0.33, 'said Mayan': 0.3, 'named unrelated language': 0.015, 'declined': 0.005, 'other branch': 0.005} |
| Sakapulteko | Kaqchikel, K'iche' | {'nearest branch': 0.543, 'named nearest relative': 0.226, 'said Mayan': 0.192, 'said Spanish/English': 0.019, 'declined': 0.01, 'named unrelated language': 0.01} |
| Popti' (Jakalteko) | Q'anjob'al | {'named nearest relative': 0.652, 'said Mayan': 0.248, 'nearest branch': 0.043, 'declined': 0.028, 'named unrelated language': 0.021, 'other branch': 0.007} |
| Yucatec Maya | none close | {'said Mayan': 0.601, 'declined': 0.182, 'other branch': 0.095, 'named unrelated language': 0.068, 'said Spanish/English': 0.054} |
| Tseltal | none close | {'said Mayan': 0.383, 'named unrelated language': 0.376, 'declined': 0.22, 'other branch': 0.014, 'said Spanish/English': 0.007} |
| Tsotsil | none close | {'said Mayan': 0.547, 'named unrelated language': 0.216, 'declined': 0.115, 'other branch': 0.101, 'said Spanish/English': 0.02} |
| Ch'ol | none close | {'said Mayan': 0.47, 'named unrelated language': 0.289, 'declined': 0.148, 'other branch': 0.094} |
| Tojolabal | Q'anjob'al | {'said Mayan': 0.582, 'named nearest relative': 0.199, 'named unrelated language': 0.116, 'declined': 0.048, 'other branch': 0.041, 'said Spanish/English': 0.007, 'nearest branch': 0.007} |
| Huastec | none close | {'said Mayan': 0.473, 'named unrelated language': 0.393, 'said Spanish/English': 0.047, 'other branch': 0.047, 'declined': 0.04} |
| Chuj | Q'anjob'al | {'said Mayan': 0.492, 'said Spanish/English': 0.186, 'named unrelated language': 0.153, 'named nearest relative': 0.119, 'other branch': 0.034, 'declined': 0.017} |
