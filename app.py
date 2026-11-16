"""Which Interpreter? - live demo.

Speak (or upload) a few seconds of speech; the tool suggests which interpreter to request
(Spanish, Mam, K'iche', Q'eqchi', Kaqchikel or Q'anjob'al), says English needs none, or says the
language is not supported / it is not sure. The decision rule is wi.decide, the same code that
wi.validate scores, so the demo behaves exactly as reported. Styling lives in app.css.

Run: make app
"""
import json
import time
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import gradio as gr
import soundfile as sf
import torch
import torchaudio.functional as AF

from wi.config import CACHE, LANGS, MAYAN, MODELS, NAMES, RESULTS, SR
from wi.decide import clip_logprobs, decide, windows
from wi.embed import Embedder
from wi.experiments import split_new_corpora
from wi.phones import PhoneRecognizer

router = joblib.load(MODELS / "router.joblib")
osm = joblib.load(MODELS / "openset.joblib")
emb, rec = Embedder(), PhoneRecognizer()
VOCAB = router["phone_vectorizer"].get_feature_names_out()
LOG = MODELS.parent / "logs" / "demo_predictions.jsonl"
OWN = MODELS.parent / "data" / "own_voice"

ROUTE = {l: "spa" if l == "spa" else "eng" if l == "eng" else "mayan" for l in LANGS}
COLOR = {"spa": "var(--wi-blue)", "mayan": "var(--wi-jade)", "eng": "var(--wi-slate)"}
SOURCE = {"tachiwin": "radio", "coerll": "course audio", "tzijonel": "podcast", "google_es": "read speech",
          "librispeech": "audiobook", "bloom": "storybook", "fleurs": "read sentences",
          "omnilingual": "community recording", "chuj": "field recording"}

ARROW = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 9.5h11V4l9 8-9 8v-5.5H2z" fill="currentColor"/></svg>'
CHECK = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.5 12.5l5.5 5.5L20.5 6.5" fill="none" '
         'stroke="currentColor" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"/></svg>')
ALERT = ('<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" '
         'stroke-width="2.6"/><path d="M12 6.5v7" stroke="currentColor" stroke-width="2.8" stroke-linecap="round"/>'
         '<circle cx="12" cy="17.3" r="1.6" fill="currentColor"/></svg>')

HEADER = f"""
<header class="wi-head">
  <div class="wi-mark" style="color:#fff">{ARROW}</div>
  <h1>Which Interpreter?</h1>
  <p>For clinic and legal-aid intake: record a few short answers and it suggests which interpreter to call,
  so Mayan-speaking patients aren't sent a Spanish interpreter by default.</p>
  <ul class="wi-routes" aria-label="Languages it covers">
    <li style="--c:var(--wi-blue)">Spanish</li>
    <li style="--c:var(--wi-jade)">Mam · K'iche' · Q'eqchi' · Kaqchikel · Q'anjob'al</li>
    <li style="--c:var(--wi-slate)">English <span class="lbl">(no interpreter needed)</span></li>
    <li style="--c:var(--wi-caution)"><span class="lbl">Anything else: says so</span></li>
  </ul>
</header>"""

FOOT = ("<div class='wi-foot'><strong>Prototype for interpreter access only.</strong> It suggests which interpreter "
        "to call; a person always confirms with the speaker. It must not be used to decide where someone is from, "
        "or in immigration or asylum decisions. Trained mostly on read-aloud storybooks and a few broadcast and "
        "podcast sources, so accuracy on stressed, conversational or phone speech is lower. Audio is not stored "
        "unless you tick “Save” for your own voice.</div>")


def stephead(n, title, help_text=""):
    h = f"<div class='wi-stephead'><b>{n}</b><h2>{title}</h2></div>"
    return h + (f"<p class='wi-help'>{help_text}</p>" if help_text else "")


def sign(cls, eyebrow, icon, main, sub):
    return (f"<div class='wi-sign wi-sign--{cls}' role='status'><div class='wi-sign__eyebrow'>{eyebrow}</div>"
            f"<div class='wi-sign__main'>{icon}<span>{main}</span></div><div class='wi-sign__sub'>{sub}</div></div>")


EMPTY = ("<div class='wi-result'>" + sign("empty", "Waiting for audio", "", "Record an answer, then press Identify language.",
         "The interpreter to call appears here, with how the tool decided.") + "</div>")


def multi_answer_note():
    """Quote the multi-answer benefit measured for THIS pipeline (wi.validate), if available."""
    f = RESULTS / "validation.json"
    if not f.exists():
        return ""
    v = json.loads(f.read_text())["internal"]
    rows = {n: [r for r in v[n] if r["language"] in ("Mam", "K'iche'", "Kaqchikel")] for n in ("1", "3")}
    if not rows["1"] or not rows["3"]:
        return ""
    m1 = np.mean([r["useful"] for r in rows["1"]]); m3 = np.mean([r["useful"] for r in rows["3"]])
    return (f" On held-out Mam, K'iche' and Kaqchikel recordings, useful answers went from {m1:.0%} "
            f"with one answer to {m3:.0%} with three.")


NOTE = multi_answer_note()


def to_16k_mono(audio):
    sr, x = audio
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(1)
    if np.abs(x).max() > 1.5:          # int16 from the microphone
        x = x / 32768.0
    if sr != SR:
        x = AF.resample(torch.from_numpy(x)[None], sr, SR)[0].numpy()
    return x


def sound_patterns(phone_strings, lang, k=6):
    """Phone n-grams in this audio that most pushed the phone classifier toward `lang`
    (tf-idf value x that class's weight relative to the mean weight over the 7 classes)."""
    texts = [p for p in phone_strings if p]
    if not texts:
        return []
    x = np.asarray(router["phone_vectorizer"].transform(texts).mean(0)).ravel()
    W = router["phone_clf"].coef_
    c = x * (W[list(router["phone_clf"].classes_).index(lang)] - W.mean(0))
    return [VOCAB[j] for j in np.argsort(c)[::-1][:k] if c[j] > 0]


def pct(p, d=0):
    """Percent that never claims certainty: '>99%' / '<1%' at the ends."""
    if p >= 1 - 0.5 * 10 ** -(d + 2):
        return f">{100 - 10 ** -d:g}%"
    if 0 < p < 0.5 * 10 ** -(d + 2):
        return f"<{10 ** -d:g}%"
    return f"{100 * p:.{d}f}%"


def lane(name, p, line, color, win=False, total=False):
    cls = "wi-lane" + (" wi-lane--win" if win else "") + (" wi-lane--total" if total else "")
    return (f"<div class='{cls}' style='--c:{color}'><span class='wi-lane__name'>{escape(name)}</span>"
            f"<div class='wi-bar' aria-hidden='true'><i style='width:{100 * p:.1f}%'></i>"
            f"<b style='left:{100 * line:.1f}%'></b></div><span class='wi-lane__val'>{pct(p)}</span></div>")


def verdict(d, n_answers):
    """(sign HTML, note HTML, plain-text headline for the log)."""
    kind, lang, conf = d["kind"], d["lang"], d["conf"]
    basis = f"from {n_answers} answer{'s' if n_answers > 1 else ''}"
    note = ""
    if kind == "named" and lang == "eng":
        head = ("eng", "English", CHECK, "No interpreter needed", f"{pct(conf)} sure, {basis}.")
    elif kind == "named":
        n = escape(NAMES[lang])
        head = (ROUTE[lang], "Call this interpreter", ARROW, n, f"Request a <strong>{n}</strong> interpreter. {pct(conf)} sure, {basis}.")
    elif kind == "mayan_try_two":
        a, b = (escape(NAMES[l]) for l in lang)
        head = ("mayan", "A Mayan language, not Spanish", ARROW, f"Try {a}, then {b}",
                f"Request a <strong>{a}</strong> interpreter first, then <strong>{b}</strong>. Close Mayan languages are "
                "much harder to tell apart from short clips than Mayan is from Spanish.")
    elif kind == "not supported":
        head = ("caution", "Stop and check", ALERT, "Not a supported language",
                "Use the picture-based language card, or ask the interpreter service to identify the language.")
        note = ("If the person is from Guatemala or southern Mexico, it may be a Mayan language this tool doesn't cover, "
                "such as Akateko, Chuj or Ixil. This check also turns away much unscripted Spanish recorded unlike our "
                "training audio, so if the person may speak Spanish, ask them.")
    else:
        head = ("caution", "Not sure", ALERT, "Ask for another answer",
                f"The audio may be unclear or in a language it doesn't cover. Top guess: "
                f"<strong>{escape(NAMES[d['top']])}</strong>, {pct(conf)}.")
    cls, eyebrow, icon, main, sub = head
    plain = f"{eyebrow}: {main}".replace("&#x27;", "'")
    return sign(cls, eyebrow, icon, main, sub), (f"<div class='wi-note'>{note}</div>" if note else ""), plain


def trace(d):
    """Signature panel: the two checks wi.decide makes, with the line each score has to reach."""
    p7, ks, thr = d["p7"], d["known_score"], osm["threshold"]
    tau, tau_m, kind = router["tau"], router["tau_mayan"], d["kind"]
    passed = kind != "not supported"
    c1 = (f"<div class='wi-check'><div class='wi-check__q'><span>Check 1</span>Is it one of the seven languages?</div>"
          + lane("Supported", ks, thr, "var(--wi-ink)" if passed else "var(--wi-caution)", win=True)
          + f"<p class='wi-check__a'>{'Yes' if passed else 'No'}: scores {pct(ks, 1)}, "
            f"{'above' if passed else 'below'} the {thr:.1%} line. An “other language” class trained on 131 "
            "languages decides this.</p></div>")
    wins = set([d["lang"]] if kind == "named" else d["lang"] if kind == "mayan_try_two" else [])
    lanes = [lane(NAMES["spa"], p7[0], tau, COLOR["spa"], "spa" in wins)]
    fam = "".join(lane(NAMES[l], p7[LANGS.index(l)], tau, COLOR["mayan"], l in wins) for l in MAYAN)
    fam += lane("Any Mayan", d["p_mayan"], tau_m, COLOR["mayan"], kind == "mayan_try_two", total=True)
    lanes.append(f"<div class='wi-family'><div class='wi-family__label'>Mayan languages</div>{fam}</div>")
    lanes.append(lane(NAMES["eng"], p7[LANGS.index("eng")], tau, COLOR["eng"], "eng" in wins))
    if kind == "named":
        a2 = f"{escape(NAMES[d['lang']])} reaches the {tau:.0%} line, so it names one language."
    elif kind == "mayan_try_two":
        a2 = (f"No single language reaches {tau:.0%}, but Mayan languages together reach {pct(d['p_mayan'])} "
              f"(line {tau_m:.0%}), so it names the top two.")
    elif kind == "not sure":
        a2 = (f"No language reaches {tau:.0%}, and Mayan languages together reach only {pct(d['p_mayan'])} "
              f"(line {tau_m:.0%}), so it says it isn't sure.")
    else:
        a2 = (f"Skipped, because check 1 said no. Closest guess: {escape(NAMES[d['top']])}, {pct(d['conf'])}.")
    c2 = (f"<div class='wi-check{'' if passed else ' wi-check--skipped'}'><div class='wi-check__q'><span>Check 2</span>"
          f"Which one?</div><div class='wi-lanes'>{''.join(lanes)}</div><p class='wi-check__a'>{a2}</p>"
          "<div class='wi-legend'><b></b>the line a score must reach. Scores blend a language-fingerprint model and "
          "a speech-sound model.</div></div>")
    return f"<div class='wi-trace'><h3>How it decided</h3>{c1}{c2}</div>"


def sounds(d, phone_strings):
    heard = " │ ".join(p for p in phone_strings if p)
    if not heard:
        return ""
    target = d["lang"][0] if d["kind"] == "mayan_try_two" else d["top"]
    # A declined clip has no language to explain; showing "patterns toward English" for French misleads.
    pats = [] if d["kind"] == "not supported" else sound_patterns(phone_strings, target)
    chips = "".join(f"<code>{escape(p)}</code>" for p in pats)
    lead = (f"Sound patterns that pointed the speech-sound model toward <strong>{escape(NAMES[target])}</strong>"
            + (", its top guess" if d["kind"] == "not sure" else "") + ":")
    return (f"<div class='wi-sounds'><h3>Sounds it heard</h3>"
            + (f"<p>{lead}</p><div class='wi-chips wi-chips--{ROUTE[target]}'>{chips}</div>" if pats else "")
            + f"<details><summary>Everything it heard, in phonetic letters (IPA)</summary>"
            f"<p class='wi-ipa'>{escape(heard[:600])}</p></details></div>")


def predict(*args):
    *audios, save, spoken = args
    chunks, n_answers = [], 0
    for a in audios:
        if a is not None:
            x = to_16k_mono(a)
            if len(x) >= SR:
                chunks += windows(x); n_answers += 1
    if not chunks:
        return ("<div class='wi-result'>" + sign("caution", "No usable audio", ALERT, "Record a longer answer",
                "Record or upload at least one answer of 2–3 seconds or more, then press Identify language.") + "</div>")
    H, _ = emb(chunks)
    phone_strings = rec(chunks)
    L7, L8 = clip_logprobs(H, phone_strings, router, osm)
    d = decide(L7, L8, router, osm)
    scores = {NAMES[l]: float(v) for l, v in zip(LANGS, d["p7"])}
    sign_html, note_html, plain = verdict(d, n_answers)
    n = d["n_clips"]
    meta = (f"Based on {n_answers} answer{'s' if n_answers > 1 else ''} ({n} clip{'s' if n > 1 else ''}).")
    if n_answers < 3:
        meta += " Another answer usually helps with recording types it knows (not with unfamiliar ones)." + NOTE
    html = (f"<div class='wi-result'>{sign_html}{note_html}<div class='wi-meta'>{meta}</div>"
            f"{trace(d)}{sounds(d, phone_strings)}</div>")
    stamp, files = time.strftime("%Y%m%d-%H%M%S"), []
    if save:                                    # own voice only (hackathon rules)
        OWN.mkdir(parents=True, exist_ok=True)
        tag = "".join(c for c in (spoken or "unknown").lower() if c.isalnum()) or "unknown"
        for i, a in enumerate(audios):
            if a is not None:
                f = OWN / f"{stamp}_{tag}_{i}.wav"
                sf.write(f, to_16k_mono(a), SR); files.append(f.name)
    with open(LOG, "a") as fh:                  # decisions only, never audio
        fh.write(json.dumps({"time": stamp, "spoken": spoken, "decision": plain,
                             "kind": d["kind"], "known_score": d["known_score"], "threshold": osm["threshold"],
                             "scores": scores, "n_clips": d["n_clips"], "phones": " | ".join(phone_strings)[:300],
                             "saved": files}) + "\n")
    return html


def examples_table():
    """Only clips that neither shipped model trained on: held-out halves of the new corpora, held-out
    English books, and open-set TEST languages (the shipped open-set model trained on 'train' languages
    only). Excluded (duplicate/jingle) clips are skipped."""
    man = pd.read_parquet(MODELS.parent / "data" / "manifest.parquet")
    _, test_groups = split_new_corpora(man.assign(role=man.role))
    held = man[((man.role == "xcorpus") & man.group.isin(test_groups) & man.corpus.ne("fleurs")) |
               (man.role == "heldout_eng")]
    picks = held.groupby("label", group_keys=False).apply(lambda g: g.sample(min(4, len(g)), random_state=0))
    unseen = man[(man.os_split.fillna("") == "test") & man.role.ne("excluded")]
    unseen = unseen[unseen.label.isin(["fra", "por", "pol", "jac", "cac", "tzj", "poc", "quv"])]
    picks = pd.concat([picks.assign(kind="held-out"),
                       unseen.groupby("label", group_keys=False).apply(lambda g: g.head(2)).assign(kind="unseen language")])
    out = RESULTS / "examples"; out.mkdir(exist_ok=True)
    for cid in picks.clip_id:
        f = out / f"{cid}.wav"
        if not f.exists():
            sf.write(f, np.load(CACHE / f"{cid}.npy").astype(np.float32), SR)
    picks["file"] = picks.clip_id.map(lambda c: str(out / f"{c}.wav"))
    picks["language"] = picks.label.map(lambda l: NAMES.get(l, {"pol": "Polish"}.get(l, l)))
    return picks


def example_labels(df):
    """'Mam · radio 1', 'Mam · radio 2', ... in display order."""
    base = [f"{r.language} · {SOURCE.get(r.corpus, r.corpus)}" for r in df.itertuples()]
    seen = {}
    return [f"{b} {seen.setdefault(b, []).append(0) or len(seen[b])}" if base.count(b) > 1 else b for b in base]


EX = examples_table()
ORDER = {l: i for i, l in enumerate(LANGS)}
KNOWN = EX[EX.kind == "held-out"].sort_values("label", key=lambda s: s.map(ORDER), kind="stable")
UNSEEN = EX[EX.kind == "unseen language"]
UNSEEN = pd.concat([UNSEEN[UNSEEN.label.isin(["fra", "por", "pol"])], UNSEEN[~UNSEEN.label.isin(["fra", "por", "pol"])]])

THEME = gr.themes.Base(
    font=[gr.themes.Font("Atkinson Hyperlegible"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.Font("Noto Sans Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#EEF1EC", body_text_color="#18222B", body_text_color_subdued="#56605A",
    background_fill_secondary="#F6F8F5", color_accent="#0B6E55", color_accent_soft="#DCEDE6",
    border_color_primary="#D5DBD3", border_color_accent="#0B6E55",
    block_background_fill="#FFFFFF", block_border_color="#D5DBD3", block_border_width="1px",
    block_radius="14px", block_shadow="none", block_label_background_fill="#FFFFFF",
    block_label_text_color="#18222B", block_label_text_weight="700", block_label_border_width="0px",
    block_title_text_color="#18222B", block_title_text_weight="700",
    panel_background_fill="#FFFFFF", panel_border_color="#D5DBD3",
    input_background_fill="#FFFFFF", input_border_color="#D5DBD3", input_border_color_focus="#1F4FB8",
    checkbox_background_color_selected="#0B6E55", checkbox_border_color_selected="#0B6E55",
    checkbox_label_background_fill="#FFFFFF", checkbox_label_border_color="#D5DBD3",
    button_large_radius="12px", button_medium_radius="10px", button_small_radius="8px",
    button_primary_background_fill="#18222B", button_primary_background_fill_hover="#0B6E55",
    button_primary_text_color="#FFFFFF", button_primary_border_color="#18222B",
    button_primary_border_color_hover="#0B6E55",
    button_secondary_background_fill="#FFFFFF", button_secondary_background_fill_hover="#F6F8F5",
    button_secondary_border_color="#D5DBD3", button_secondary_text_color="#18222B",
    loader_color="#0B6E55", link_text_color="#1F4FB8",
)
HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700'
        '&family=Noto+Sans+Mono:wght@400;600&family=Overpass:wght@700;800;900&display=swap">')
LIGHT = """() => { const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u.href); } }"""
WAVE = gr.WaveformOptions(waveform_color="#9AA79F", waveform_progress_color="#0B6E55")


def answer(label):
    return gr.Audio(sources=["microphone", "upload"], type="numpy", label=label, waveform_options=WAVE)


with gr.Blocks(title="Which Interpreter?") as demo:
    gr.HTML(HEADER, padding=False, apply_default_css=False)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=340):
            gr.HTML(stephead(1, "Record up to three answers",
                             "Ask the person to answer in the language they speak at home: their name, their town, "
                             "what brings them in. 3–10 seconds each."), padding=False, apply_default_css=False)
            audio = answer("Answer 1 · name")
            audio2 = answer("Answer 2 · town (optional)")
            audio3 = answer("Answer 3 · reason for visit (optional)")
            with gr.Row():
                clear = gr.Button("Clear answers", variant="secondary", scale=1)
                btn = gr.Button("Identify language", variant="primary", scale=2, elem_id="wi-identify")
            with gr.Accordion("Save test recordings (your own voice only)", open=False):
                save = gr.Checkbox(label="Save these recordings for testing", value=False)
                spoken = gr.Textbox(label="Language you spoke (optional)", placeholder="e.g. French")
        with gr.Column(scale=6, min_width=340):
            gr.HTML(stephead(2, "See which interpreter to call"), padding=False, apply_default_css=False)
            result = gr.HTML(EMPTY, padding=False, apply_default_css=False)
    btn.click(predict, [audio, audio2, audio3, save, spoken], [result])
    clear.click(lambda: (None, None, None, EMPTY), outputs=[audio, audio2, audio3, result])
    gr.HTML("<div class='wi-section'><h2>Try clips it never trained on</h2><p>Click one to load it into Answer 1, "
            "then press Identify language. None of these recordings, and none of the bottom row's languages, "
            "were used to train either model.</p></div>", padding=False, apply_default_css=False)
    gr.Examples(examples=[[f] for f in KNOWN.file], inputs=audio, label="Supported languages, new recordings",
                example_labels=example_labels(KNOWN), examples_per_page=len(KNOWN))
    gr.Examples(examples=[[f] for f in UNSEEN.file], inputs=audio,
                label="Languages it should decline (Mayan relatives are the hardest)",
                example_labels=example_labels(UNSEEN), examples_per_page=len(UNSEEN))
    gr.HTML(FOOT, padding=False, apply_default_css=False)

if __name__ == "__main__":
    demo.launch(theme=THEME, css_paths=[Path(__file__).with_name("app.css")], head=HEAD, js=LIGHT,
                footer_links=["gradio"], run_history=False)
