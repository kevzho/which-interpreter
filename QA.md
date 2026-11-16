# Finals Q&A Cheat Sheet

## If you only remember three things (30 seconds)

1. **The harmful mistake is sending a Mayan speaker to a Spanish interpreter.** Meta's MMS model does that 24% of the time on unseen storybooks. Ours: 4.0%, and 0–2% on held-out recordings from series we adapted to.
2. **We only trust tests on things the model never saw.** A random split scores 0.98; held-out books score 0.79. The 0.18 gap is memorisation, so 0.79 is our number.
3. **Our biggest failure is the "I don't know" check.** In the blind test it turned away 87–91% of radio and parliament Spanish, though the classifier's top guess was right for 93–99% of those clips. It learned what our recordings sound like. Fix: recalibrate on a few clips at each clinic.

## If the live demo fails

> "That's the exact failure we measured: new mic, new room. It says 'Stop and check' and tells staff to ask if the person speaks Spanish, and 'How it decided' still shows its closest guess. Here's a clip the model never trained on." Click a built-in example (runs offline).

## Top 15 questions (most likely and hardest first)

**1. Why didn't you test Spanish live? / What if it says "not supported" for a Spanish speaker?**
None of us speaks Spanish, and a learner's or synthetic accent isn't what a patient sounds like, so we demo our own English live and show held-out Spanish clips. We did measure synthetic Spanish: 14 text-to-speech voices were named Spanish in only 4 of 42 answers, though the classifier underneath picked Spanish in 49 of the 51 declined runs (`results/synthetic_voices.md`). It's the same failure as our blind test, which turned away 87–91% of radio and parliament Spanish. The screen says "Stop and check", tells staff to ask if the person speaks Spanish, and still shows its closest guess under check 2, so it costs time, not a wrong interpreter.
*Don't say:* "It works on any Spanish," or blame the mic.

**2. Your check rejects Spanish the classifier got right. Why? Wouldn't deploying it make things worse?**
The "other" class heard 131 languages from many sources, but our Spanish came from a few read-aloud sources, so "sounds unfamiliar" became "probably other." Even audiobook Spanish was 36% rejected, and adding more read Spanish didn't help (Argentine Spanish stayed about 88%). Today it would slow Spanish speakers down, but it refuses rather than misroutes. Fix: reset the threshold on a few dozen consented clips per site.
*Don't say:* "It's just unscripted speech" (audiobooks are scripted), or "we fixed it."

**3. A random split gives 98%. Why trust 0.79? Isn't it memorising speakers?**
The 98% is the leak: clips from one book land on both sides. Holding out whole books drops macro-F1 to 0.79. Narrators can still repeat, so we also tested sources it never trained on, where the storybook-only model got only about 20–42%. We froze our models with SHA-256 hashes before scoring the blind tests.
*Don't say:* "No leakage" or "speaker-independent" (Bloom has no speaker labels).

**4. Close relatives like Tz'utujil are declined only 22–44% of the time. Don't errors land on the smallest communities?**
Yes, that's our second failure. They're close cousins of K'iche' and Kaqchikel, so distant "other" languages can't teach that boundary; farther-off Chuj and Popti' get declined about 94%. Most of the rest get "try two" or one of our Mayan languages; at most 1.5% were sent to Spanish (4 of 264 Poqomam clips) and none to English. The real fix is recordings of those languages, plus native speakers telling us if a K'iche' interpreter actually helps.
*Don't say:* "A K'iche' interpreter can serve them" (never tested).

**5. Could a border agency use this to check where someone is from? What do you store?**
Language isn't nationality: Mam is spoken in Guatemala and Mexico, and the tool only names an interpreter. It's also a bad origin test, since it mixes up close relatives. Audio is saved only if someone ticks a box; the log keeps decisions, scores and a short phone string, which we'd drop for a clinic. A disclaimer isn't a lock, so we'd release it only to clinics and legal-aid groups.
*Don't say:* "It can't be misused" or "we store nothing."

**6. You say three answers beat one. What if the model is confidently wrong?**
It gets more confidently wrong. On held-out K'iche', three answers raised useful answers from 82% to 92%. On blind radio Spanish they fell from 13% to 4%, because one speaker on one mic has the same bias every time, and averaging only cancels random noise.
*Don't say:* "More answers always help."

**7. Is your "I don't know" method really better than plain max-probability?**
Partly. On AUROC (1.0 perfect, 0.5 coin flip) it's clearly better on 31 distant test languages: 0.99 versus 0.90, gain interval +0.03 to +0.17. Across all 37 the gain isn't significant, and on the 6 Mayan relatives it's worse, 0.83 versus 0.89. We chose the method on 26 separate tune languages first.
*Don't say:* "It helps with Mayan relatives," or quote old v1 numbers.

**8. Why these languages, and how many people would this help?**
Intake staff often can't tell a Mayan language from Spanish, and Bloom had rare recorded speech in five Guatemalan Mayan languages. The one number we can cite: in DOJ's FY2016 immigration-court data, Mam ranked 9th and K'iche' 10th of all languages, ahead of French (EOIR Statistics Yearbook, Table 7). We didn't estimate a clinic head count. We measured misrouting instead: always-Spanish 100%, MMS 24%, ours 4.0%. Languages with no data, like Akateko and Ixil, get nothing from us yet.
*Don't say:* any population number you can't cite, or that the ranking is current (it's FY2016).

**9. Did the Mayan speakers consent? Did any community have a say?**
No community was consulted. We used public datasets under their licences, dropped no-derivatives Bloom books, and accepted SIL's terms, which cite the UN Declaration on Indigenous rights. Weak spots: the Mexican radio audio has no stated licence, and half the no-derivatives Kaqchikel podcast trained our model. Next: paid, consented recordings through an interpreter agency, with a Mayan language organisation controlling the data.
*Don't say:* "All our data is consented."

**10. Which mistake is most expensive, and can I trust the confidence?**
Mayan to a Spanish interpreter. When unsure, the tool says "Mayan, try X then Y" or "not sure," never Spanish; on held-out recordings it named languages 97–100% correctly and sent 0–2% of Mayan to Spanish. Under new sources confidence breaks: the storybook-only model named a language for 5.8% of new K'iche' clips and was right on only 16.7% of those.
*Don't say:* "It's calibrated."

**11. How would a real clinic use it, and how would you prove it helps?**
A worker records three short answers, a person confirms, and "not sure" falls back to the picture language card. We'd launch only Spanish, English, Mam, K'iche' and Kaqchikel; Q'eqchi' and Q'anjob'al have no test audio outside storybooks. Run in shadow mode: interpreters record the real language, we recalibrate on part, freeze, score the rest once. If it doesn't beat today on Mayan-to-Spanish misroutes, we stop.
*Don't say:* that any clinic has tested it.

**12. Why logistic regression on frozen features, not fine-tuning? Are your baselines weak?**
Data is small and the 0.18 leakage gap showed how easily models memorise, so we chose a small model we could retrain hundreds of times. Zero-shot MMS is a strong baseline that already knows all five Mayan languages: 0.55 macro-F1 and 24% to Spanish, versus our 0.79 and 4.0%. We never tried fine-tuning, so we can't say it's worse.
*Don't say:* "Fine-tuning would overfit."

**13. A probe tells your storybook Spanish from FLEURS Spanish perfectly. Did removing that shortcut help?**
Not enough to keep, and we report that. The probe scores AUROC 1.000 on the fingerprint features (0.988 on speech sounds). Erasing recording-source information with LEACE left K'iche' on a left-out source at 8.9%, no better than erasing random directions (8.0%). A stronger projection (removing the top 200 between-recording directions) lifted it to 24.1% but cut Google Spanish from 98.6% to 93.2%, and picking how much to remove would have meant tuning on the test sources, so we didn't ship it. What worked was data: about 10 recordings from a new source raised accuracy there from 20–42% to 81–88%.
*Don't say:* "The model ignores recording conditions."

**14. MMS already covers these languages. What's new?**
The parts aren't new: frozen embeddings, phone n-grams, outlier exposure. What's new is the task, routing five Guatemalan Mayan languages, Spanish and English with a "try two" fallback and a "not supported" answer, plus a reusable test with held-out books, sources and languages and hash-frozen blind tests. MMS has no "none of these" answer.
*Don't say:* "First-ever Mayan language ID."

**15. How fast is it, and what hardware does a clinic need?**
It runs offline on this laptop, so patient audio never leaves the device. One answer takes about 0.2 seconds and three about 0.7 seconds on a 2024 MacBook (Apple M4), after a ~30-second model load at startup. We'd benchmark clinic hardware before a pilot.
*Don't say:* "Real-time" or "runs on a phone."

**16. Mam is your demo language, but its storybook recall is 46%, below MMS's 60%. Why?**
One book. 1,068 of the 1,540 Mam storybook clips (69%) come from *Pujb'il Yol Mam*; when it's held out, the model learns Mam from only about 470 clips in 16 other books and gets 32% on it, while the other Mam books score 75–98%. The misses stay Mayan (K'iche' 20%, Q'anjob'al 20%, Q'eqchi' 6%, Kaqchikel 6%); 1.6% go to Spanish. On held-out Mam radio, the full app names Mam or "try Mam" 89% of the time and sends none to Spanish. (Recomputed with the same held-out-book folds as `report.md`.)
*Don't say:* "Mam is our best language."

**17. Isn't MMS's 0.55 unfair, since it was never offered English?**
Yes, English scores 0 for MMS; on the other six languages it gets at least 0.64. That's why our headline is the Mayan-to-Spanish rate (24% vs 4.0%), which doesn't involve English at all.
*Don't say:* "MMS is bad."

**18. Why not just ask which town they're from?**
Staff would need to know which of Guatemala's 20-plus Mayan languages is spoken where, many towns are bilingual, and in an immigration setting a "where are you from" question can frighten people. The tool listens to the language itself and never asks or guesses origin.
*Don't say:* that the tool can tell where someone is from.

**19. Interpreter phone lines already help identify languages. Why is this better?**
We didn't measure phone-line vendors, so we don't claim to beat them. The tool is a first step before the call: in under a second, offline, it suggests which language line to ask for, and says "not sure" when it can't tell.
*Don't say:* "Vendors get it wrong."

**20. Walk us through the 50/50 blend and the thresholds.**
Each classifier gives probabilities for the 7 languages; we average their log-probabilities with equal weight (picked on held-out storybooks, which is why the blended 0.80 counts as in-sample). Check 1: the "other language" model's "supported" score must reach 95.8%, a line set so that 90% of held-out known-language clips pass. Check 2: name a language at 85% or more; otherwise "try two" if all Mayan languages together reach 83%; otherwise "not sure". The app draws exactly these lines (`wi/decide.py`).

**21. What did your team do, and what did Claude do?**
[Team: answer in your own words, truthfully. The README says Claude Code wrote much of the code and ran experiments under your direction. Before finals, each member should be able to explain `wi/decide.py` (the decision rule), `wi/openset.py` (the "other language" class and its threshold), the held-out splits in `wi/experiments.py`, and why the models were frozen before the blind tests.]
*Don't say:* anything that understates the AI's role.

## Fix before finals

- Fill question 21 in your own words before finals.
- Record own-voice tests with "Save" ticked (English, plus any unsupported language you speak, several takes each), then run `make ownvoice` and put the result in the README's failure list. If a Spanish-speaking family member or friend agrees, record them too (the brief allows a family member's voice with permission).
- Q15 timings measured: 0.16 s / 0.65 s.
- Done: the README's "at most 1.5%" (full app) is correct; `shipped_model.md`'s 3.0% is the router alone, before the "not supported" check, and is now labelled that way.
