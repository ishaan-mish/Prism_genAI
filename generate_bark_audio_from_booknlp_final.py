# generate_full_sequential_wavs_entities_bark_gender_v2.py
# Bark TTS + improved gender override/inference logic that uses:
#  - numeric / name / regex overrides,
#  - mention_phrase from quotes file,
#  - entity pronoun evidence from .entities,
#  - optional gender-guesser,
#  - heuristics and fallbacks.
#
# Outputs gender_report.json for auditing.

import os
import re
import csv
import math
import json
import traceback
from io import BytesIO
import pandas as pd
from pydub import AudioSegment
import sys
import numpy as np

# ---------- optional gender_guesser ----------
try:
    from gender_guesser.detector import Detector as GenderDetector
    _GG_AVAILABLE = True
    _GG_DETECTOR = GenderDetector()
except Exception:
    _GG_AVAILABLE = False
    _GG_DETECTOR = None

# ---------- bark imports ----------
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
except Exception as e:
    print("Failed to import bark. Make sure 'bark' is installed. Error:", e)
    raise

# ---------- CONFIG ----------
BASE_DIR = r"C:\Users\golec\Downloads\final fantasy\VOICOVERS"
QUOTES_FILE = os.path.join(BASE_DIR, "a_study_in_scarlet_actuak_8npuys.quotes")
ENTITIES_FILE = os.path.join(BASE_DIR, "a_study_in_scarlet_actuak_8npuys.entities")
SUPERSENSE_FILE = os.path.join(BASE_DIR, "a_study_in_scarlet_actuak_8npuys.supersense")  # optional
FULL_TEXT_FILE = os.path.join(BASE_DIR, "full_text.txt")
OUT_DIR = os.path.join(BASE_DIR, "wavs")
MANIFEST_CSV = os.path.join(BASE_DIR, "manifest.csv")
GENDER_REPORT_JSON = os.path.join(BASE_DIR, "gender_report.json")
os.makedirs(OUT_DIR, exist_ok=True)

# Base voice profiles (tweak pitch/speed per character id if desired)
VOICE_PROFILES = {
    "narrator": {"speed": 1.00, "pitch": 0},
    0:  {"speed": 1.00, "pitch": 0},
    20: {"speed": 0.94, "pitch": -2},
    23: {"speed": 1.06, "pitch": 2},
    70: {"speed": 0.90, "pitch": -4},
    85: {"speed": 1.12, "pitch": 3},
    91: {"speed": 1.00, "pitch": 1},
}

# Improved override map:
# Keys may be:
#   - int cluster ids (20: 'male')
#   - exact character name strings ("Sherlock Holmes": "male")
#   - regex patterns prefixed with 're:' ("re:^dr\\s+watson": "male")
GENDER_OVERRIDES = {
    20: "male",                      # numeric id (Sherlock)
    "Dr. John Watson": "male",       # exact name match
    "re:^dr\\s+watson": "male",      # regex for name patterns (case-insensitive)
    # add more as needed
}

# Gender pitch tweak
GENDER_PITCH_OFFSET = {"male": -2, "female": 2}

# TTS chunk sizes and silence padding
TTS_CHUNK_SIZE = 240
CHUNK_GAP_MS = 100

# Bark options (tweak as desired)
BARK_TEMP = 0.7
BARK_HISTORY_PROMPTS = {"male": None, "female": None, "narrator": None}
# -----------------------------------

# ---------- Helpers ----------
def safe_read_quotes(path):
    # Read tab-separated quotes file; keep mention_phrase column if present.
    try:
        df = pd.read_csv(path, sep="\t", engine="python", quoting=3, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, sep="\t", engine="python", encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    # remove header-like garbage rows where the 'quote' cell contains header text
    if "quote" in df.columns:
        df = df[~df["quote"].astype(str).str.lower().str.strip().isin(["quote", "qotes", "quote_start", "quote_end"])]
    # drop empty quotes
    df = df[df["quote"].notna()]
    # normalize whitespace and strip stray quotes
    if "quote" in df.columns:
        df["quote"] = df["quote"].astype(str).apply(lambda s: re.sub(r'^[\s"\u201c\u201d\']+|[\s"\u201c\u201d\']+$', '', s).strip())
    # numeric starts/ends
    for col in ("quote_start","quote_end","mention_start","mention_end","char_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    # keep mention_phrase as-is (string)
    if "mention_phrase" in df.columns:
        df["mention_phrase"] = df["mention_phrase"].astype(str).fillna("").apply(lambda s: s.strip())
    return df

def load_entities(path):
    """
    Load .entities file and return mapping:
      cluster_id -> {'text':..., 'prop':..., 'cat':..., 'raw_row':Series}
    Be robust to header (COREF) lines and different column counts.
    """
    mapping = {}
    if not os.path.exists(path):
        print("Entities file not found:", path)
        return mapping
    # Try to read flexible tab-separated file; some files include header, some don't.
    try:
        ent = pd.read_csv(path, sep="\t", header=0, engine="python", encoding="utf-8")
        # if header exists and the first column is non-numeric label like "COREF", try to coerce columns
        cols = [c.strip() for c in ent.columns]
        ent.columns = cols
        # If header contains 'COREF' rename to 'cluster' for compatibility
        if 'COREF' in ent.columns:
            ent = ent.rename(columns={'COREF':'cluster'})
    except Exception:
        ent = pd.read_csv(path, sep="\t", header=None, engine="python", encoding="utf-8")

    # Ensure we have at least first and last columns: cluster ... text
    # Heuristic: last column is text if columns > 1
    for _, r in ent.iterrows():
        try:
            cluster_val = r.iloc[0]
            # try int cluster
            cid = int(cluster_val)
        except Exception:
            # skip rows with non-numeric cluster id
            continue
        # text likely in last column
        txt = ""
        try:
            txt = str(r.iloc[-1]).strip()
        except Exception:
            txt = ""
        # try to find prop/cat if present
        prop = ""
        cat = ""
        if 'prop' in ent.columns:
            prop = str(r.get('prop', '')).strip()
        if 'cat' in ent.columns:
            cat = str(r.get('cat', '')).strip()
        mapping[cid] = {"text": txt, "prop": prop, "cat": cat, "raw_row": r}
    return mapping

def _compile_override_maps(override_raw):
    id_overrides = {}
    name_overrides = {}
    regex_overrides = []
    for k, v in override_raw.items():
        gender = None if v is None else str(v).lower()
        if isinstance(k, int):
            id_overrides[k] = gender
        elif isinstance(k, str):
            if k.startswith("re:"):
                pat = k[3:]
                try:
                    regex_overrides.append((re.compile(pat, re.I), gender))
                except re.error:
                    print("Invalid regex override pattern:", pat)
            else:
                name_overrides[k.strip().lower()] = gender
        else:
            try:
                ki = int(k)
                id_overrides[ki] = gender
            except Exception:
                name_overrides[str(k).strip().lower()] = gender
    return id_overrides, name_overrides, regex_overrides

_ID_OVERRIDES, _NAME_OVERRIDES, _REGEX_OVERRIDES = _compile_override_maps(GENDER_OVERRIDES)

def _map_gender_from_gender_guesser(name):
    if not _GG_AVAILABLE or not name:
        return None
    first = name.strip().split()[0]
    if not first:
        return None
    try:
        g = _GG_DETECTOR.get_gender(first)
    except Exception:
        return None
    if g in ("male", "mostly_male"):
        return "male"
    if g in ("female", "mostly_female"):
        return "female"
    return None

def infer_gender_from_name(name):
    """Heuristic name/title checks and fuzzy suffixes."""
    if not name or not isinstance(name, str):
        return None
    n = name.lower().strip()
    # titles / honorifics
    if re.search(r'\b(mrs|miss|ms|lady|madam|madame|queen)\b', n):
        return "female"
    if re.search(r'\b(mr|sir|lord|king)\b', n):
        return "male"
    # fuzzy female endings
    if re.search(r'(ina|ette|elle|ine|essa|ara|ia)$', n):
        return "female"
    # pronoun indicators in the string
    if re.search(r'\b(she|her|hers|woman|female)\b', n):
        return "female"
    if re.search(r'\b(he|him|his|man|male)\b', n):
        return "male"
    # last resort: gender_guesser
    gg = _map_gender_from_gender_guesser(n)
    if gg:
        return gg
    return None

def build_entity_pronoun_evidence(entities_map):
    """
    Inspect entity texts: if an entity text is a clear pronoun (he/she/him/her),
    record that as strong evidence: cluster -> 'male'|'female'
    """
    evidence = {}
    for cid, info in entities_map.items():
        txt = (info.get("text") or "").strip().lower()
        if txt in ("he", "him", "his"):
            evidence[cid] = ("male", "entity_pronoun")
        elif txt in ("she", "her", "hers"):
            evidence[cid] = ("female", "entity_pronoun")
        # also check prop/cat hints (cat==PRON/PER may not suffice, but if text contains pronoun words we used above)
    return evidence

def resolve_gender_for_row(cid, char_text, mention_phrase, entities_map, entity_pronoun_evidence):
    """
    Resolve gender for a single char (used by per-quote logic). Returns (gender or None, reason).
    Priority:
      1) numeric id override
      2) exact name override
      3) regex override
      4) mention_phrase (from quotes row) if present and determinative
      5) entity pronoun evidence (from .entities)
      6) gender_guesser on entity text
      7) heuristics on name/title
      8) None (caller may fallback to default)
    """
    # 1: id override
    try:
        if isinstance(cid, int) and cid in _ID_OVERRIDES:
            return _ID_OVERRIDES[cid], "override_id"
    except Exception:
        pass

    # normalised candidate name
    name_norm = (str(char_text) or "").strip()
    name_norm_lower = name_norm.lower()

    # 2: exact name override
    if name_norm_lower in _NAME_OVERRIDES:
        return _NAME_OVERRIDES[name_norm_lower], "override_name"

    # 3: regex overrides
    for pat, gender in _REGEX_OVERRIDES:
        if pat.search(name_norm):
            return gender, "override_regex"

    # 4: mention_phrase evidence (from quotes row) - strong signal if pronoun
    if mention_phrase:
        mp = mention_phrase.strip().lower()
        # common direct pronouns
        if re.search(r'\b(he|him|his)\b', mp):
            return "male", "mention_phrase"
        if re.search(r'\b(she|her|hers)\b', mp):
            return "female", "mention_phrase"
        # possessive/patterns: 'my companion' - ambiguous; try to link cluster evidence:
        if re.search(r'\b(my companion|companion)\b', mp):
            # See if same cluster has entity pronoun evidence
            if isinstance(cid, int) and cid in entity_pronoun_evidence:
                return entity_pronoun_evidence[cid][0], "mention_phrase+entity_link"
            # otherwise uncertain (let fallthrough decide)
    # 5: entity pronoun evidence
    if isinstance(cid, int) and cid in entity_pronoun_evidence:
        return entity_pronoun_evidence[cid][0], entity_pronoun_evidence[cid][1]

    # 6: gender_guesser on entity text
    gg = _map_gender_from_gender_guesser(name_norm)
    if gg:
        return gg, "gender_guesser"

    # 7: heuristics on name/title
    h = infer_gender_from_name(name_norm)
    if h:
        return h, "heuristic_name"

    return None, "unknown"

# chunking / bark TTS / audio helpers (same as before)
def chunk_text_to_sentences(text, max_chars=TTS_CHUNK_SIZE):
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    pieces = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks = []
    for p in pieces:
        if len(p) <= max_chars:
            chunks.append(p.strip())
        else:
            subparts = re.split(r'(?<=,)\s+', p)
            tmp = ""
            for s in subparts:
                if len(tmp) + len(s) + 1 <= max_chars:
                    tmp = (tmp + " " + s).strip() if tmp else s.strip()
                else:
                    if tmp:
                        chunks.append(tmp.strip())
                    if len(s) <= max_chars:
                        tmp = s.strip()
                    else:
                        for i in range(0, len(s), max_chars):
                            chunks.append(s[i:i+max_chars].strip())
                        tmp = ""
            if tmp:
                chunks.append(tmp.strip())
    # join small trailing pieces
    final = []
    for c in chunks:
        if not final:
            final.append(c)
        else:
            if len(final[-1]) + len(c) + 1 <= max_chars:
                final[-1] = final[-1] + " " + c
            else:
                final.append(c)
    return final

def bark_text_to_segment(text, history_prompt=None, temp=BARK_TEMP):
    try:
        wav = generate_audio(text, history_prompt=history_prompt, temp=temp)
        if not isinstance(wav, np.ndarray):
            wav = np.asarray(wav, dtype=np.float32)
        wav = np.clip(wav, -1.0, 1.0)
        pcm = (wav * 32767.0).astype(np.int16)
        seg = AudioSegment(data=pcm.tobytes(), sample_width=2, frame_rate=int(SAMPLE_RATE), channels=1)
        return seg
    except TypeError:
        # fallback variants of generate_audio signature
        try:
            wav = generate_audio(text)
            wav = np.asarray(wav, dtype=np.float32)
            wav = np.clip(wav, -1.0, 1.0)
            pcm = (wav * 32767.0).astype(np.int16)
            seg = AudioSegment(data=pcm.tobytes(), sample_width=2, frame_rate=int(SAMPLE_RATE), channels=1)
            return seg
        except Exception as e:
            print("Bark fallback TTS failed:", e)
            return AudioSegment.silent(duration=400)
    except Exception as e:
        print("Bark TTS failed for chunk (len={}): {}".format(len(text), e))
        return AudioSegment.silent(duration=400)

def apply_speed_pitch(seg, speed=1.0, pitch=0):
    if not math.isclose(speed, 1.0, rel_tol=1e-6):
        new_rate = int(seg.frame_rate * float(speed))
        seg = seg._spawn(seg.raw_data, overrides={"frame_rate": new_rate})
        seg = seg.set_frame_rate(44100)
    if pitch != 0:
        new_rate = int(seg.frame_rate * (2.0 ** (pitch / 12.0)))
        seg = seg._spawn(seg.raw_data, overrides={"frame_rate": new_rate})
        seg = seg.set_frame_rate(44100)
    return seg

# ---------- Main ----------
def main():
    # preload Bark models (may download/cache on first run)
    try:
        preload_models()
    except Exception as e:
        print("Warning: preload_models() failed or is slow. Continuing — Bark will attempt to load models on demand. Error:", e)

    # check files
    if not os.path.exists(QUOTES_FILE):
        print("Missing quotes file:", QUOTES_FILE); return
    if not os.path.exists(FULL_TEXT_FILE):
        print("Missing full text file:", FULL_TEXT_FILE); return

    quotes_df = safe_read_quotes(QUOTES_FILE).sort_values(by="quote_start").reset_index(drop=True)
    entities_map = load_entities(ENTITIES_FILE)
    entity_pronoun_evidence = build_entity_pronoun_evidence(entities_map)

    # Build initial char_gender map using improved resolver
    char_gender = {}
    char_gender_reasons = {}

    # First pass: use entities_map entries (if any) to populate base genders
    for cid, info in entities_map.items():
        char_text = info.get("text", "")
        g, reason = resolve_gender_for_row(cid, char_text, None, entities_map, entity_pronoun_evidence)
        if g:
            char_gender[cid] = g
            char_gender_reasons[cid] = reason

    # Then use quote-level mention_phrase to refine or add genders (high priority per-row)
    for _, row in quotes_df.iterrows():
        cid_raw = row.get("char_id", "")
        try:
            cid = int(cid_raw)
        except Exception:
            cid = cid_raw
        mention_phrase = row.get("mention_phrase", "")
        # Try to find a textual char name from entities_map or row
        char_name = ""
        if isinstance(cid, int) and cid in entities_map:
            char_name = entities_map[cid].get("text", "")
        else:
            char_name = str(row.get("char_name", "")) or str(row.get("speaker", "")) or ""

        g, reason = resolve_gender_for_row(cid, char_name, mention_phrase, entities_map, entity_pronoun_evidence)
        if g:
            # prefer mention_phrase derived gender over prior weaker guesses
            prev = char_gender.get(cid)
            if prev is None or reason in ("mention_phrase", "mention_phrase+entity_link", "override_id", "override_name", "override_regex"):
                char_gender[cid] = g
                char_gender_reasons[cid] = reason

    # Apply any explicit numeric overrides that might not map to an entity
    for k, v in _ID_OVERRIDES.items():
        char_gender[k] = v
        char_gender_reasons[k] = "override_id_explicit"

    # final fallback for any unresolved clusters: heuristics or default to 'male' for compatibility
    for _, row in quotes_df.iterrows():
        cid_raw = row.get("char_id", "")
        try:
            cid = int(cid_raw)
        except Exception:
            cid = cid_raw
        if cid not in char_gender:
            # try to infer from char_name if present
            char_name = ""
            if isinstance(cid, int) and cid in entities_map:
                char_name = entities_map[cid].get("text", "")
            else:
                char_name = str(row.get("char_name", "")) or str(row.get("speaker", "")) or ""
            g, reason = resolve_gender_for_row(cid, char_name, row.get("mention_phrase", ""), entities_map, entity_pronoun_evidence)
            if g:
                char_gender[cid] = g
                char_gender_reasons[cid] = reason
            else:
                # fallback default - preserve previous script behaviour (male) but mark as fallback
                char_gender[cid] = "male"
                char_gender_reasons[cid] = "fallback_default_male"

    # Write a gender report (audit)
    report = {}
    for k, v in char_gender.items():
        report[str(k)] = {"gender": v, "reason": char_gender_reasons.get(k, "unknown")}
    with open(GENDER_REPORT_JSON, "w", encoding="utf-8") as jf:
        json.dump(report, jf, indent=2, ensure_ascii=False)

    print("Gender report written to:", GENDER_REPORT_JSON)
    print("Resolved genders sample (first 20):", dict(list(report.items())[:20]))

    # Now continue with TTS generation (same as before)
    manifest = []
    seq = 1
    pos = 0  # current char position in full text
    full_text = open(FULL_TEXT_FILE, "r", encoding="utf-8").read()
    narrator_gender = "female"

    for _, row in quotes_df.iterrows():
        try:
            qs = int(row.get("quote_start", 0))
            qe = int(row.get("quote_end", 0))
        except Exception:
            qs = 0; qe = 0

        # NARRATOR chunk before quote
        narrator_text = full_text[pos:qs].strip()
        if narrator_text:
            chunks = chunk_text_to_sentences(narrator_text)
            for c in chunks:
                seg = bark_text_to_segment(c, history_prompt=BARK_HISTORY_PROMPTS.get("narrator"))
                base = VOICE_PROFILES.get("narrator", {"speed":1.0,"pitch":0})
                gender_off = GENDER_PITCH_OFFSET.get(narrator_gender, 0)
                seg = apply_speed_pitch(seg, speed=base["speed"], pitch=base["pitch"] + gender_off)
                seg = seg + AudioSegment.silent(CHUNK_GAP_MS)
                fname = f"{seq:04d}.wav"
                seg.export(os.path.join(OUT_DIR, fname), format="wav")
                manifest.append({"seq":seq, "filename":fname, "type":"narrator", "char_id":"", "char_name":"narrator", "text":c})
                seq += 1

        # QUOTE chunk(s)
        quote_text = str(row.get("quote","")).strip()
        if quote_text and quote_text.lower() not in ("quote","qotes","quote_start","quote_end"):
            quote_text = re.sub(r'^(quote|qotes)[:\s\-]*', '', quote_text, flags=re.I).strip()
            cid_raw = row.get("char_id", "")
            try:
                cid = int(cid_raw)
            except Exception:
                cid = cid_raw
            # char name resolution
            char_name = ""
            if isinstance(cid, int) and cid in entities_map:
                char_name = entities_map[cid].get("text", "")
            else:
                char_name = str(row.get("char_name", "")) or str(row.get("speaker", "")) or ""

            gender = char_gender.get(cid, None)
            if not gender:
                # fallback to heuristics if somehow unresolved
                g, reason = resolve_gender_for_row(cid, char_name, row.get("mention_phrase", ""), entities_map, entity_pronoun_evidence)
                gender = g or "male"
                char_gender[cid] = gender
                char_gender_reasons[cid] = reason

            base = VOICE_PROFILES.get(cid, VOICE_PROFILES.get("narrator"))
            gender_off = GENDER_PITCH_OFFSET.get(gender, 0)

            history_prompt = BARK_HISTORY_PROMPTS.get(gender if gender in BARK_HISTORY_PROMPTS else None)

            chunks = chunk_text_to_sentences(quote_text)
            for c in chunks:
                seg = bark_text_to_segment(c, history_prompt=history_prompt)
                seg = apply_speed_pitch(seg, speed=base["speed"], pitch=base["pitch"] + gender_off)
                seg = seg + AudioSegment.silent(CHUNK_GAP_MS)
                fname = f"{seq:04d}.wav"
                seg.export(os.path.join(OUT_DIR, fname), format="wav")
                manifest.append({"seq":seq, "filename":fname, "type":"quote", "char_id":cid, "char_name":char_name, "text":c})
                seq += 1

        pos = qe

    # trailing narrator
    trailing = full_text[pos:].strip()
    if trailing:
        chunks = chunk_text_to_sentences(trailing)
        for c in chunks:
            seg = bark_text_to_segment(c, history_prompt=BARK_HISTORY_PROMPTS.get("narrator"))
            base = VOICE_PROFILES.get("narrator", {"speed":1.0,"pitch":0})
            seg = apply_speed_pitch(seg, speed=base["speed"], pitch=base["pitch"] + GENDER_PITCH_OFFSET.get(narrator_gender,0))
            seg = seg + AudioSegment.silent(CHUNK_GAP_MS)
            fname = f"{seq:04d}.wav"
            seg.export(os.path.join(OUT_DIR, fname), format="wav")
            manifest.append({"seq":seq, "filename":fname, "type":"narrator", "char_id":"", "char_name":"narrator", "text":c})
            seq += 1

    # write manifest CSV
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as mf:
        w = csv.DictWriter(mf, fieldnames=["seq","filename","type","char_id","char_name","text"])
        w.writeheader()
        for r in manifest:
            w.writerow(r)

    print("Done: generated", seq-1, "wav files in", OUT_DIR)
    print("Manifest saved to", MANIFEST_CSV)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal:", e)
        traceback.print_exc()
        sys.exit(1)
