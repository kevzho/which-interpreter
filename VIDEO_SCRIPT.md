# Demo video script (2:00)

Screen recording of the app (`make app`, open http://127.0.0.1:7860) with voice-over. At ~2.5 spoken words per second each section fits its slot. The brief says a screen recording with voice-over is plenty; trim waits if you edit, and caption any cut "sped up".

**One main takeaway:** on storybooks it never heard, ours sent 4% of Mayan clips to a Spanish interpreter (Meta's model used as-is: 24%), and when it isn't sure, it says so.

**Before recording:**
- Close other apps; the demo uses ~4 GB. Wait ~30 s after `make app` for the models to load. Zoom the browser to 125% so "How it decided" is readable.
- Clips used below (all held out; checked in our run of all 36 examples): "Mam · radio 1" → Mam; "Spanish · read speech 3" → Stop and check (Spanish 97% underneath); "French · storybook 1" → declined. Avoid "K'iche' · course audio 2" and "Kaqchikel · podcast 1" (declined).
- The opening audio is `results/examples/tach_Campaña Convenio 169 - A18 Bullet - Salud - Mam_seg12.mp3.wav` (the same clip as "Mam · radio 1"). macOS screen recording doesn't capture app audio, so either add the first 2 s of that file in editing, or play it through the speakers with your mic on.
- Own voice: in the app, tick "Save these recordings for testing", type "English", record Answer 1 ("My name is …, I live in …, and I'm here because …", 5–8 s) and press Identify. It's saved to `data/own_voice/`. The video uses your first take, whatever it says.

## 1. The problem (0:00–0:15, ~33 words + 2 s of audio)
*On screen: the app's header. Captions: "Mam: 9th of all languages in US immigration court cases, FY2016 (DOJ EOIR, Table 7)" and "Interpreter access only. Not for origin, immigration or asylum decisions."*
[2 s of the Mam clip] "Would your front desk call a Spanish interpreter for this? It's Mam, from Guatemala: in 2016, the ninth most common language in US immigration courts. Which Interpreter? tells staff who to call."

## 2. The demo (0:15–1:15, ~105 words + clicks)
*Before each Identify, caption the truth ("Correct: Mam") so viewers see our answer next to it.*
1. **Mam** (click "Mam · radio 1", Identify). "That's the clip you just heard, from a radio recording it never trained on." → *Call this interpreter: Mam.* Point at "How it decided": "Check one: is it one of our seven languages? Yes, above the line. Check two: Mam crosses the 85 percent line, so it names Mam."
2. **Our biggest weakness** (click "Spanish · read speech 3", Identify). "This is Spanish. Underneath, it's 97 percent sure, but check one lands just below the line, so it says 'stop and check' instead of calling any interpreter."
3. **French** (click "French · storybook 1", Identify). "French isn't one of our languages. Check one fails, and that's correct."
4. **My own voice** (caption: "My own voice, recorded today"). Your English answer, saved from the app beforehand, goes into Answer 1 → *No interpreter needed.* "That's my own voice in English: no interpreter needed." Don't read Spanish as a non-speaker: text-to-speech Spanish was declined 51 of 56 times.

## 3. How it works (1:15–1:44, ~72 words)
*On screen: one bar chart, "Mayan clips sent to a Spanish interpreter (storybooks it never trained on)": always Spanish 100%, Meta MMS used as-is 24%, ours 4%.*
"About ten thousand storybook clips in seven languages, plus radio, a podcast and course audio. Two of Meta's speech models, used as-is, turn each clip into a language fingerprint and a list of speech sounds; small classifiers on top make the two checks you saw. On storybooks it never heard, ours sent 4 percent of Mayan clips to a Spanish interpreter; Meta's model used as-is sent 24. And when it isn't sure, it says so."

## 4. Next steps (1:44–2:00, ~40 words)
"Biggest weakness: about nine in ten radio and parliament Spanish clips got 'stop and check'; almost none got a wrong interpreter. For now, it's a second opinion when Spanish isn't working. Next: retune with consenting speakers at a real intake desk."

---
Upload to YouTube as **Unlisted**, then open the link in a private window to check it plays without signing in.
