# Synthetic-voice test (macOS text-to-speech)

Frozen shipped models, same decision rule as the demo, scored once; nothing trained or tuned on these. Each voice reads three intake answers (name, town, reason for visit); we score each answer alone and all three combined. 'Useful' = the right interpreter (Spanish), 'no interpreter needed' (English), or declining (languages the tool doesn't support). Synthetic speech is a recording condition absent from our training data.

| language | status | voices | useful, 1 answer | useful, 3 answers | what it said (1 answer) |
|---|---|---|---|---|---|
| Spanish | supported | 14 | 4/42 | 1/14 | {'Not a supported language': 38, 'Spanish interpreter': 4} |
| English | supported | 7 | 21/21 | 7/7 | {'English: no interpreter': 21} |
| French | never heard | 3 | 9/9 | 3/3 | {'Not a supported language': 9} |
| Portuguese | never heard | 3 | 9/9 | 3/3 | {'Not a supported language': 9} |
| Mandarin | used only to set the line | 3 | 9/9 | 3/3 | {'Not a supported language': 9} |

Spanish declined by the "not supported" check: 51 runs; the language classifier underneath still had Spanish as its top guess in 49 of them. This is the same failure as the blind tests: the check treats unfamiliar-sounding Spanish as another language.

## Every run

| language | voice | answers | what it said | top guess underneath |
|---|---|---|---|---|
| Spanish | Paulina | 1 | Spanish interpreter | Spanish |
| Spanish | Paulina | 1 | Spanish interpreter | Spanish |
| Spanish | Paulina | 1 | Spanish interpreter | Spanish |
| Spanish | Paulina | 3 | Spanish interpreter | Spanish |
| Spanish | Mónica | 1 | Not a supported language | Spanish |
| Spanish | Mónica | 1 | Spanish interpreter | Spanish |
| Spanish | Mónica | 1 | Not a supported language | Spanish |
| Spanish | Mónica | 3 | Not a supported language | Spanish |
| Spanish | Eddy (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Eddy (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Eddy (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Eddy (Spanish (Mexico)) | 3 | Not a supported language | Spanish |
| Spanish | Eddy (Spanish (Spain)) | 1 | Not a supported language | Q'anjob'al |
| Spanish | Eddy (Spanish (Spain)) | 1 | Not a supported language | Spanish |
| Spanish | Eddy (Spanish (Spain)) | 1 | Not a supported language | Spanish |
| Spanish | Eddy (Spanish (Spain)) | 3 | Not a supported language | Spanish |
| Spanish | Flo (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Flo (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Flo (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Flo (Spanish (Mexico)) | 3 | Not a supported language | Spanish |
| Spanish | Flo (Spanish (Spain)) | 1 | Not a supported language | Spanish |
| Spanish | Flo (Spanish (Spain)) | 1 | Not a supported language | Spanish |
| Spanish | Flo (Spanish (Spain)) | 1 | Not a supported language | Spanish |
| Spanish | Flo (Spanish (Spain)) | 3 | Not a supported language | Spanish |
| Spanish | Grandma (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Grandma (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Grandma (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Grandma (Spanish (Mexico)) | 3 | Not a supported language | Spanish |
| Spanish | Grandma (Spanish (Spain)) | 1 | Not a supported language | Spanish |
| Spanish | Grandma (Spanish (Spain)) | 1 | Not a supported language | Spanish |
| Spanish | Grandma (Spanish (Spain)) | 1 | Not a supported language | Spanish |
| Spanish | Grandma (Spanish (Spain)) | 3 | Not a supported language | Spanish |
| Spanish | Grandpa (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Grandpa (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Grandpa (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Grandpa (Spanish (Mexico)) | 3 | Not a supported language | Spanish |
| Spanish | Grandpa (Spanish (Spain)) | 1 | Not a supported language | Spanish |
| Spanish | Grandpa (Spanish (Spain)) | 1 | Not a supported language | Spanish |
| Spanish | Grandpa (Spanish (Spain)) | 1 | Not a supported language | Spanish |
| Spanish | Grandpa (Spanish (Spain)) | 3 | Not a supported language | Spanish |
| Spanish | Reed (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Reed (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Reed (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Reed (Spanish (Mexico)) | 3 | Not a supported language | Spanish |
| Spanish | Reed (Spanish (Spain)) | 1 | Not a supported language | Q'anjob'al |
| Spanish | Reed (Spanish (Spain)) | 1 | Not a supported language | Spanish |
| Spanish | Reed (Spanish (Spain)) | 1 | Not a supported language | Spanish |
| Spanish | Reed (Spanish (Spain)) | 3 | Not a supported language | Spanish |
| Spanish | Rocko (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Rocko (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Rocko (Spanish (Mexico)) | 1 | Not a supported language | Spanish |
| Spanish | Rocko (Spanish (Mexico)) | 3 | Not a supported language | Spanish |
| Spanish | Rocko (Spanish (Spain)) | 1 | Not a supported language | Spanish |
| Spanish | Rocko (Spanish (Spain)) | 1 | Not a supported language | Spanish |
| Spanish | Rocko (Spanish (Spain)) | 1 | Not a supported language | Spanish |
| Spanish | Rocko (Spanish (Spain)) | 3 | Not a supported language | Spanish |
| English | Samantha | 1 | English: no interpreter | English |
| English | Samantha | 1 | English: no interpreter | English |
| English | Samantha | 1 | English: no interpreter | English |
| English | Samantha | 3 | English: no interpreter | English |
| English | Eddy (English (US)) | 1 | English: no interpreter | English |
| English | Eddy (English (US)) | 1 | English: no interpreter | English |
| English | Eddy (English (US)) | 1 | English: no interpreter | English |
| English | Eddy (English (US)) | 3 | English: no interpreter | English |
| English | Flo (English (US)) | 1 | English: no interpreter | English |
| English | Flo (English (US)) | 1 | English: no interpreter | English |
| English | Flo (English (US)) | 1 | English: no interpreter | English |
| English | Flo (English (US)) | 3 | English: no interpreter | English |
| English | Grandma (English (US)) | 1 | English: no interpreter | English |
| English | Grandma (English (US)) | 1 | English: no interpreter | English |
| English | Grandma (English (US)) | 1 | English: no interpreter | English |
| English | Grandma (English (US)) | 3 | English: no interpreter | English |
| English | Grandpa (English (US)) | 1 | English: no interpreter | English |
| English | Grandpa (English (US)) | 1 | English: no interpreter | English |
| English | Grandpa (English (US)) | 1 | English: no interpreter | English |
| English | Grandpa (English (US)) | 3 | English: no interpreter | English |
| English | Reed (English (US)) | 1 | English: no interpreter | English |
| English | Reed (English (US)) | 1 | English: no interpreter | English |
| English | Reed (English (US)) | 1 | English: no interpreter | English |
| English | Reed (English (US)) | 3 | English: no interpreter | English |
| English | Rocko (English (US)) | 1 | English: no interpreter | English |
| English | Rocko (English (US)) | 1 | English: no interpreter | English |
| English | Rocko (English (US)) | 1 | English: no interpreter | English |
| English | Rocko (English (US)) | 3 | English: no interpreter | English |
| French | Jacques | 1 | Not a supported language | English |
| French | Jacques | 1 | Not a supported language | English |
| French | Jacques | 1 | Not a supported language | English |
| French | Jacques | 3 | Not a supported language | English |
| French | Eddy (French (France)) | 1 | Not a supported language | English |
| French | Eddy (French (France)) | 1 | Not a supported language | English |
| French | Eddy (French (France)) | 1 | Not a supported language | English |
| French | Eddy (French (France)) | 3 | Not a supported language | English |
| French | Flo (French (France)) | 1 | Not a supported language | English |
| French | Flo (French (France)) | 1 | Not a supported language | English |
| French | Flo (French (France)) | 1 | Not a supported language | English |
| French | Flo (French (France)) | 3 | Not a supported language | English |
| Portuguese | Luciana | 1 | Not a supported language | Spanish |
| Portuguese | Luciana | 1 | Not a supported language | Spanish |
| Portuguese | Luciana | 1 | Not a supported language | Spanish |
| Portuguese | Luciana | 3 | Not a supported language | Spanish |
| Portuguese | Eddy (Portuguese (Brazil)) | 1 | Not a supported language | English |
| Portuguese | Eddy (Portuguese (Brazil)) | 1 | Not a supported language | Spanish |
| Portuguese | Eddy (Portuguese (Brazil)) | 1 | Not a supported language | English |
| Portuguese | Eddy (Portuguese (Brazil)) | 3 | Not a supported language | Spanish |
| Portuguese | Flo (Portuguese (Brazil)) | 1 | Not a supported language | Spanish |
| Portuguese | Flo (Portuguese (Brazil)) | 1 | Not a supported language | Spanish |
| Portuguese | Flo (Portuguese (Brazil)) | 1 | Not a supported language | Spanish |
| Portuguese | Flo (Portuguese (Brazil)) | 3 | Not a supported language | Spanish |
| Mandarin | Eddy (Chinese (China mainland)) | 1 | Not a supported language | Q'anjob'al |
| Mandarin | Eddy (Chinese (China mainland)) | 1 | Not a supported language | English |
| Mandarin | Eddy (Chinese (China mainland)) | 1 | Not a supported language | Kaqchikel |
| Mandarin | Eddy (Chinese (China mainland)) | 3 | Not a supported language | Q'anjob'al |
| Mandarin | Flo (Chinese (China mainland)) | 1 | Not a supported language | Kaqchikel |
| Mandarin | Flo (Chinese (China mainland)) | 1 | Not a supported language | English |
| Mandarin | Flo (Chinese (China mainland)) | 1 | Not a supported language | Kaqchikel |
| Mandarin | Flo (Chinese (China mainland)) | 3 | Not a supported language | Kaqchikel |
| Mandarin | Reed (Chinese (China mainland)) | 1 | Not a supported language | Q'anjob'al |
| Mandarin | Reed (Chinese (China mainland)) | 1 | Not a supported language | English |
| Mandarin | Reed (Chinese (China mainland)) | 1 | Not a supported language | Q'anjob'al |
| Mandarin | Reed (Chinese (China mainland)) | 3 | Not a supported language | Q'anjob'al |
