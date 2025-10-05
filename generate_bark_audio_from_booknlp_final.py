# generate_bark_audio_robust.py
# Robust pipeline:
# - robust full_text reading (UTF-8 / cp1252 fallback)
# - optional tokens -> byte offset mapping to handle token-indexed quotes
# - snap quote boundaries to word boundaries if needed
# - incremental manifest (written as each WAV is exported)
# - stable narrator voice (history prompt + low temp)
# - Bark TTS output -> pydub, per-character voice adjustments
# - gender resolution / report

import os, re, csv, math, json, traceback, sys
from io import BytesIO
from pathlib import Path
import pandas as pd
import numpy as np
from pydub import AudioSegment

# optional gender_guesser
try:
    from gender_guesser.detector import Detector as GenderDetector
    _GG_AVAILABLE = True
    _GG_DETECTOR = GenderDetector()
except Exception:
    _GG_AVAILABLE = False
    _GG_DETECTOR = None

# bark imports
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
except Exception as e:
    print("Error importing bark. Install bark and dependencies. Error:", e)
    raise

# ---------------- CONFIG (edit these paths if needed) ----------------
BASE_DIR = r"C:\Users\golec\Downloads\final fantasy\VOICOVERS"   # path containing .quotes/.entities/full_text.txt
QUOTES_FILE = os.path.join(BASE_DIR, "a_study_in_scarlet_actuak_8npuys.quotes")
ENTITIES_FILE = os.path.join(BASE_DIR, "a_study_in_scarlet_actuak_8npuys.entities")
FULL_TEXT_FILE = os.path.join(BASE_DIR, "full_text.txt")
TOKENS_FILE = os.path.join(BASE_DIR, "tokens.tsv")  # optional; if not present, script falls back
OUT_DIR = os.path.join(BASE_DIR, "wavs")
MANIFEST_CSV = os.path.join(BASE_DIR, "manifest.csv")
GENDER_REPORT_JSON = os.path.join(BASE_DIR, "gender_report.json")
os.makedirs(OUT_DIR, exist_ok=True)

# voice/pitch profiles (tweak to taste)
VOICE_PROFILES = {
    "narrator": {"speed": 1.00, "pitch": 0},
    0:  {"speed": 1.00, "pitch": 0},
    20: {"speed": 0.94, "pitch": -2},
    23: {"speed": 1.06, "pitch": 2},
    70: {"speed": 0.90, "pitch": -4},
    85: {"speed": 1.12, "pitch": 3},
    91: {"speed": 1.00, "pitch": 1},
}

# gender overrides: numeric id, exact name, or regex ("re:pattern")
GENDER_OVERRIDES = {
    20: "male",
}

GENDER_PITCH_OFFSET = {"male": -2, "female": 2}

# chunking / silence
TTS_CHUNK_SIZE = 240
CHUNK_GAP_MS = 100

# Bark temps & prompts: stable narrator
BARK_TEMP = 0.7
NARRATOR_TEMP = 0.0
BARK_HISTORY_PROMPTS = {
    "male": None,
    "female": None,
    "narrator": "Narration voice: calm, neutral, steady pacing, clear enunciation."
}

# debug options
DEBUG_FIRST_N = 0  # set >0 to dump first N mapped quote -> substring examples and exit

# ---------------- Helper functions ----------------
def read_full_text_robust(path):
    """Read full_text with encoding fallback and common mojibake fixes. Returns bytes and decoded text and encoding used."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # try to read bytes first (we will use bytes when mapping by token byte offsets)
    b = open(path, "rb").read()
    # try decode guesses
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    text = None
    used = None
    for e in encodings_to_try:
        try:
            text = b.decode(e)
            used = e
            break
        except Exception:
            continue
    if text is None:
        # fallback: decode with utf-8 with replace
        text = b.decode("utf-8", errors="replace")
        used = "utf-8-replace"
    # common mojibake fixes
    fixes = {
        "â€™": "'", "â€œ": '"', "â€�": '"', "â€”": "—", "â€“": "–",
        "Ã©": "é", "\r\n": "\n", "\r": "\n"
    }
    for a,bad in fixes.items():
        if bad in text:
            text = text.replace(bad, fixes[bad]) if False else text  # placeholder to keep pattern (we'll do proper replacements next)
    # Actually replace known mojibake pairs conservatively:
    replacements = {
        "â€™": "'", "â€œ": '"', "â€�": '"', "â€”": "—", "â€“": "–", "Ã©": "é"
    }
    for bad, good in replacements.items():
        if bad in text:
            text = text.replace(bad, good)
    text = text.replace("\x00", "")
    return b, text, used

def load_tokens_byte_mapping(tokens_path):
    """
    Load token file and return mapping token_index -> (byte_onset, byte_offset).
    Accepts tokens TSV with column names that include 'token_ID' and 'byte_onset'/'byte_offset' as in your sample.
    """
    if not os.path.exists(tokens_path):
        return {}
    try:
        tdf = pd.read_csv(tokens_path, sep="\t", engine="python", encoding="utf-8")
    except Exception:
        tdf = pd.read_csv(tokens_path, sep="\t", engine="python", encoding="cp1252", errors="replace")
    mapping = {}
    cols = [c.lower() for c in tdf.columns]
    # try common header names
    possible_id_cols = [c for c in tdf.columns if "token_id" in c.lower() or "token" in c.lower() and "within_document" in c.lower()] \
                       or [tdf.columns[3] if len(tdf.columns) > 3 else None]
    # find byte_onset / byte_offset columns heuristically
    onset_col = next((c for c in tdf.columns if "byte_onset" in c.lower() or "onset" in c.lower()), None)
    offset_col = next((c for c in tdf.columns if "byte_offset" in c.lower() or "offset" in c.lower()), None)
    # fallback: try known names
    if onset_col is None and "byte_onset" in cols:
        onset_col = tdf.columns[cols.index("byte_onset")]
    if offset_col is None and "byte_offset" in cols:
        offset_col = tdf.columns[cols.index("byte_offset")]
    # id col search
    id_col = None
    for c in tdf.columns:
        low = c.lower()
        if "token_id" in low or "token_id_within_document" in low or ("token" in low and "within" in low and "document" in low):
            id_col = c
            break
    if id_col is None:
        # try token_ID_within_document exact
        if "token_ID_within_document" in tdf.columns:
            id_col = "token_ID_within_document"
    # if we didn't find onset/offset but have columns by position, guess positions:
    if id_col is None or onset_col is None or offset_col is None:
        # try to guess based on your sample structure (token id at col index 3, byte_onset at 6, byte_offset at 7)
        if len(tdf.columns) >= 8:
            id_col = tdf.columns[3]
            onset_col = tdf.columns[6]
            offset_col = tdf.columns[7]
    if id_col is None or onset_col is None or offset_col is None:
        # can't map; return empty
        return {}
    for _, r in tdf.iterrows():
        try:
            tid = int(r[id_col])
            start = int(r[onset_col])
            end = int(r[offset_col])
            mapping[tid] = (start, end)
        except Exception:
            continue
    return mapping

def token_index_to_char_slice(token_idx, token_map, full_bytes, decode_enc):
    """
    Convert token index -> (char_start, char_end) on the decoded text by slicing bytes then decoding.
    Returns (start_char_idx, end_char_idx, substring)
    """
    if token_map is None or token_idx not in token_map:
        return None
    bstart, bend = token_map[token_idx]
    # ensure bounds
    bstart = max(0, min(bstart, len(full_bytes)))
    bend = max(0, min(bend, len(full_bytes)))
    # slice bytes and decode using the same encoding used for the full text
    sub = full_bytes[bstart:bend].decode(decode_enc, errors='replace')
    # compute character start by decoding prefix bytes
    # simpler: decode whole prefix and take len
    prefix = full_bytes[:bstart].decode(decode_enc, errors='replace')
    return len(prefix), len(prefix) + len(sub), sub

def snap_to_word_boundary(text, idx, direction="left"):
    L = len(text)
    if idx <= 0:
        return 0
    if idx >= L:
        return L
    # if at boundary already
    if (idx < L and text[idx].isspace()) or (idx > 0 and text[idx-1].isspace()):
        return idx
    if direction == "left":
        while idx > 0 and not text[idx-1].isspace():
            idx -= 1
        return idx
    else:
        while idx < L and not text[idx].isspace():
            idx += 1
        return idx

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

def tts_to_segment_bark(text, history_prompt=None, temp=None):
    try:
        if temp is None:
            wav = generate_audio(text, history_prompt=history_prompt, temp=BARK_TEMP)
        else:
            wav = generate_audio(text, history_prompt=history_prompt, temp=temp)
        wav = np.asarray(wav, dtype=np.float32)
        wav = np.clip(wav, -1.0, 1.0)
        pcm = (wav * 32767.0).astype(np.int16)
        seg = AudioSegment(data=pcm.tobytes(), sample_width=2, frame_rate=int(SAMPLE_RATE), channels=1)
        return seg
    except TypeError:
        try:
            wav = generate_audio(text)
            wav = np.asarray(wav, dtype=np.float32)
            wav = np.clip(wav, -1.0, 1.0)
            pcm = (wav * 32767.0).astype(np.int16)
            seg = AudioSegment(data=pcm.tobytes(), sample_width=2, frame_rate=int(SAMPLE_RATE), channels=1)
            return seg
        except Exception as e:
            print("Bark fallback error:", e)
            return AudioSegment.silent(duration=400)
    except Exception as e:
        print("Bark error:", e)
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

# Gender / overrides (same improved resolver as before)
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
                    print("Invalid regex override:", pat)
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
    if not name or not isinstance(name, str):
        return None
    n = name.lower().strip()
    if re.search(r'\b(mrs|miss|ms|lady|madam|madame|queen)\b', n):
        return "female"
    if re.search(r'\b(mr|sir|lord|king)\b', n):
        return "male"
    if re.search(r'(ina|ette|elle|ine|essa|ara|ia)$', n):
        return "female"
    if re.search(r'\b(she|her|hers|woman|female)\b', n):
        return "female"
    if re.search(r'\b(he|him|his|man|male)\b', n):
        return "male"
    gg = _map_gender_from_gender_guesser(n)
    if gg:
        return gg
    return None

def resolve_gender_for_row(cid, char_text, mention_phrase, entities_map, entity_pronoun_evidence):
    # id override
    try:
        if isinstance(cid, int) and cid in _ID_OVERRIDES:
            return _ID_OVERRIDES[cid], "override_id"
    except Exception:
        pass
    name_norm = (str(char_text) or "").strip()
    name_norm_lower = name_norm.lower()
    if name_norm_lower in _NAME_OVERRIDES:
        return _NAME_OVERRIDES[name_norm_lower], "override_name"
    for pat, gender in _REGEX_OVERRIDES:
        if pat.search(name_norm):
            return gender, "override_regex"
    if mention_phrase:
        mp = mention_phrase.strip().lower()
        if re.search(r'\b(he|him|his)\b', mp):
            return "male", "mention_phrase"
        if re.search(r'\b(she|her|hers)\b', mp):
            return "female", "mention_phrase"
        if re.search(r'\b(my companion|companion)\b', mp):
            if isinstance(cid, int) and cid in entity_pronoun_evidence:
                return entity_pronoun_evidence[cid][0], "mention_phrase+entity_link"
    if isinstance(cid, int) and cid in entity_pronoun_evidence:
        return entity_pronoun_evidence[cid][0], entity_pronoun_evidence[cid][1]
    gg = _map_gender_from_gender_guesser(name_norm)
    if gg:
        return gg, "gender_guesser"
    h = infer_gender_from_name(name_norm)
    if h:
        return h, "heuristic_name"
    return None, "unknown"

def build_entity_pronoun_evidence(entities_map):
    evidence = {}
    for cid, info in entities_map.items():
        txt = (info.get("text") or "").strip().lower()
        if txt in ("he", "him", "his"):
            evidence[cid] = ("male", "entity_pronoun")
        elif txt in ("she", "her", "hers"):
            evidence[cid] = ("female", "entity_pronoun")
    return evidence

def load_entities(path):
    mapping = {}
    if not os.path.exists(path):
        return mapping
    try:
        ent = pd.read_csv(path, sep="\t", header=0, engine="python", encoding="utf-8")
        ent.columns = [c.strip() for c in ent.columns]
        if 'COREF' in ent.columns:
            ent = ent.rename(columns={'COREF':'cluster'})
    except Exception:
        ent = pd.read_csv(path, sep="\t", header=None, engine="python", encoding="utf-8", names=None)
    for _, r in ent.iterrows():
        try:
            cluster_val = r.iloc[0]
            cid = int(cluster_val)
        except Exception:
            continue
        txt = ""
        try:
            txt = str(r.iloc[-1]).strip()
        except Exception:
            txt = ""
        prop = str(r.get('prop', '')).strip() if 'prop' in r.index else ""
        cat = str(r.get('cat', '')).strip() if 'cat' in r.index else ""
        mapping[cid] = {"text": txt, "prop": prop, "cat": cat, "raw_row": r}
    return mapping

def safe_read_quotes(path):
    try:
        df = pd.read_csv(path, sep="\t", engine="python", quoting=3, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, sep="\t", engine="python", encoding="cp1252", errors="replace")
    df.columns = [c.strip() for c in df.columns]
    if "quote" in df.columns:
        df = df[~df["quote"].astype(str).str.lower().str.strip().isin(["quote", "qotes", "quote_start", "quote_end"])]
    df = df[df["quote"].notna()]
    if "quote" in df.columns:
        df["quote"] = df["quote"].astype(str).apply(lambda s: re.sub(r'^[\s"\u201c\u201d\']+|[\s"\u201c\u201d\']+$', '', s).strip())
    for col in ("quote_start","quote_end","mention_start","mention_end","char_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "mention_phrase" in df.columns:
        df["mention_phrase"] = df["mention_phrase"].astype(str).fillna("").apply(lambda s: s.strip())
    return df

# ---------------- Main ----------------
def main():
    try:
        preload_models()
    except Exception as e:
        print("Warning: preload_models() may be slow or fail. Bark will load models on demand. Err:", e)

    if not os.path.exists(QUOTES_FILE):
        print("Missing quotes file:", QUOTES_FILE); return
    if not os.path.exists(FULL_TEXT_FILE):
        print("Missing full text file:", FULL_TEXT_FILE); return

    # read inputs
    quotes_df = safe_read_quotes(QUOTES_FILE).sort_values(by="quote_start").reset_index(drop=True)
    entities_map = load_entities(ENTITIES_FILE)
    entity_pronoun_evidence = build_entity_pronoun_evidence(entities_map)

    # read full_text robustly (bytes + decoded text)
    full_bytes, full_text, used_enc = read_full_text_robust(FULL_TEXT_FILE)
    print("Read full_text with encoding:", used_enc, "length chars:", len(full_text), "bytes:", len(full_bytes))

    # try tokens mapping (optional)
    token_map = load_tokens_byte_mapping(TOKENS_FILE) if os.path.exists(TOKENS_FILE) else {}
    if token_map:
        print("Loaded token -> byte mapping (tokens):", len(token_map), "entries")
    else:
        print("No token map loaded; will snap to word boundaries when needed.")

    # resolve genders and write gender_report ASAP
    char_gender = {}
    char_gender_reasons = {}
    for cid, info in entities_map.items():
        ctext = info.get("text", "")
        g, reason = resolve_gender_for_row(cid, ctext, None, entities_map, entity_pronoun_evidence)
        if g:
            char_gender[cid] = g
            char_gender_reasons[cid] = reason

    for _, row in quotes_df.iterrows():
        cid_raw = row.get("char_id", "")
        try:
            cid = int(cid_raw)
        except Exception:
            cid = cid_raw
        mention = row.get("mention_phrase", "")
        if isinstance(cid, int) and cid in entities_map:
            char_name = entities_map[cid].get("text", "")
        else:
            char_name = str(row.get("char_name", "")) or str(row.get("speaker", "")) or ""
        g, reason = resolve_gender_for_row(cid, char_name, mention, entities_map, entity_pronoun_evidence)
        if g:
            prev = char_gender.get(cid)
            if prev is None or reason in ("mention_phrase", "override_id", "override_name", "override_regex"):
                char_gender[cid] = g
                char_gender_reasons[cid] = reason

    for k,v in _ID_OVERRIDES.items():
        char_gender[k] = v
        char_gender_reasons[k] = "override_id_explicit"

    # fallback to heuristics or default 'male'
    for _, row in quotes_df.iterrows():
        cid_raw = row.get("char_id", "")
        try:
            cid = int(cid_raw)
        except Exception:
            cid = cid_raw
        if cid not in char_gender:
            if isinstance(cid, int) and cid in entities_map:
                ctext = entities_map[cid].get("text","")
            else:
                ctext = str(row.get("char_name","")) or str(row.get("speaker","")) or ""
            g, reason = resolve_gender_for_row(cid, ctext, row.get("mention_phrase",""), entities_map, entity_pronoun_evidence)
            if g:
                char_gender[cid] = g
                char_gender_reasons[cid] = reason
            else:
                char_gender[cid] = "male"
                char_gender_reasons[cid] = "fallback_default_male"

    # write gender report
    report = {str(k): {"gender": v, "reason": char_gender_reasons.get(k,"unknown")} for k,v in char_gender.items()}
    with open(GENDER_REPORT_JSON, "w", encoding="utf-8") as jf:
        json.dump(report, jf, indent=2, ensure_ascii=False)
    print("Wrote gender report:", GENDER_REPORT_JSON)

    # incremental manifest writer (header first)
    mf_handle = open(MANIFEST_CSV, "w", newline="", encoding="utf-8")
    mf_writer = csv.DictWriter(mf_handle, fieldnames=["seq","filename","type","char_id","char_name","gender","gender_reason","text"])
    mf_writer.writeheader()
    mf_handle.flush()
    try:
        os.fsync(mf_handle.fileno())
    except Exception:
        pass
    print("Created incremental manifest:", MANIFEST_CSV)


    # If debug mode, print first N quote -> substring mappings and exit
    if DEBUG_FIRST_N > 0:
        print("DEBUG mode: showing first", DEBUG_FIRST_N, "mappings")
        shown = 0
        for _, row in quotes_df.iterrows():
            if shown >= DEBUG_FIRST_N: break
            qs = int(row.get("quote_start",0)); qe = int(row.get("quote_end",0))
            substring = None
            # try token map conversion if token_map present
            if token_map and qs in token_map and qe in token_map:
                sstart, _, = token_index_to_char_slice(qs, token_map, full_bytes, used_enc)[:2]
            # fallback naive
            start = snap_to_word_boundary(full_text, qs, "left") if qs < len(full_text) else None
            end = snap_to_word_boundary(full_text, qe, "right") if qe < len(full_text) else None
            substring = full_text[start:end] if (start is not None and end is not None) else "<could not map>"
            print("QUOTES row:", qs, qe, "->", repr(substring[:200]))
            shown += 1
        mf_handle.close()
        return

    # generate audio
    seq = 1
    pos_char = 0  # char index into full_text
    total = len(quotes_df)
    print("Starting audio generation for", total, "quotes...")

    for i, row in quotes_df.iterrows():
        try:
            qs = int(row.get("quote_start", 0))
            qe = int(row.get("quote_end", 0))
        except Exception:
            qs = 0; qe = 0

        # Determine character offsets for qs, qe.
        # If token_map seems valid and indices present, map via token_map; else fallback to using value as char index (with snap)
        def map_index(idx, prefer_right=False):
            # try token map
            if token_map and idx in token_map:
                mapping = token_index_to_char_slice(idx, token_map, full_bytes, used_enc)
                if mapping:
                    return mapping[0]  # start char index
            # fallback: if idx within char length, use as char index
            if 0 <= idx <= len(full_text):
                return idx
            # if out of range, snap to nearest boundary (use left/right heuristics)
            if idx < 0:
                return 0
            return len(full_text)

        qs_char = map_index(qs)
        qe_char = map_index(qe)
        # snap boundaries to avoid cutting words
        qs_adj = snap_to_word_boundary(full_text, qs_char, direction="left")
        qe_adj = snap_to_word_boundary(full_text, qe_char, direction="right")

        # narrator chunk before the quote
        narrator_text = full_text[pos_char:qs_adj].strip()
        if narrator_text:
            chunks = chunk_text_to_sentences(narrator_text)
            for c in chunks:
                seg = tts_to_segment_bark(c, history_prompt=BARK_HISTORY_PROMPTS.get("narrator"), temp=NARRATOR_TEMP)
                base = VOICE_PROFILES.get("narrator", {"speed":1.0, "pitch":0})
                seg = apply_speed_pitch(seg, speed=base["speed"], pitch=base["pitch"] + GENDER_PITCH_OFFSET.get("female", 0))
                seg = seg + AudioSegment.silent(CHUNK_GAP_MS)
                fname = f"{seq:04d}.wav"
                out_path = os.path.join(OUT_DIR, fname)
                seg.export(out_path, format="wav")
                char_id = ""
                char_name = "narrator"
                gender = "female"
                gender_reason = "narrator_default"
                mf_writer.writerow({"seq":seq, "filename":out_path, "type":"narrator", "char_id":char_id, "char_name":char_name, "gender":gender, "gender_reason":gender_reason, "text":c})
                mf_handle.flush()
                try: os.fsync(mf_handle.fileno())
                except Exception: pass
                print("WROTE:", out_path)
                seq += 1

        # quote handling
        quote_text = str(row.get("quote","")).strip()
        if quote_text and quote_text.lower() not in ("quote","qotes","quote_start","quote_end"):
            quote_text = re.sub(r'^(quote|qotes)[:\s\-]*', '', quote_text, flags=re.I).strip()
            # determine char id and name
            cid_raw = row.get("char_id", "")
            try:
                cid = int(cid_raw)
            except Exception:
                cid = cid_raw
            if isinstance(cid, int) and cid in entities_map:
                char_name = entities_map[cid].get("text","")
            else:
                char_name = str(row.get("char_name","")) or str(row.get("speaker","")) or ""
            gender = char_gender.get(cid, None)
            gender_reason = char_gender_reasons.get(cid, "unknown")
            if not gender:
                g, r = resolve_gender_for_row(cid, char_name, row.get("mention_phrase",""), entities_map, entity_pronoun_evidence)
                gender = g or "male"
                gender_reason = r
                char_gender[cid] = gender
                char_gender_reasons[cid] = gender_reason
            base = VOICE_PROFILES.get(cid, VOICE_PROFILES.get("narrator"))
            gender_off = GENDER_PITCH_OFFSET.get(gender, 0)
            # choose history prompt by gender if available
            history = BARK_HISTORY_PROMPTS.get(gender, None)
            chunks = chunk_text_to_sentences(quote_text)
            for c in chunks:
                seg = tts_to_segment_bark(c, history_prompt=history, temp=BARK_TEMP)
                seg = apply_speed_pitch(seg, speed=base["speed"], pitch=base["pitch"] + gender_off)
                seg = seg + AudioSegment.silent(CHUNK_GAP_MS)
                fname = f"{seq:04d}.wav"
                out_path = os.path.join(OUT_DIR, fname)
                seg.export(out_path, format="wav")
                mf_writer.writerow({"seq":seq, "filename":out_path, "type":"quote", "char_id":cid, "char_name":char_name, "gender":gender, "gender_reason":gender_reason, "text":c})
                mf_handle.flush()
                try: os.fsync(mf_handle.fileno())
                except Exception: pass
                print("WROTE:", out_path)
                seq += 1

        # advance pos_char to end of quote (use qe_adj)
        pos_char = max(pos_char, qe_adj)

    # trailing narrator
    trailing = full_text[pos_char:].strip()
    if trailing:
        chunks = chunk_text_to_sentences(trailing)
        for c in chunks:
            seg = tts_to_segment_bark(c, history_prompt=BARK_HISTORY_PROMPTS.get("narrator"), temp=NARRATOR_TEMP)
            base = VOICE_PROFILES.get("narrator", {"speed":1.0, "pitch":0})
            seg = apply_speed_pitch(seg, speed=base["speed"], pitch=base["pitch"] + GENDER_PITCH_OFFSET.get("female", 0))
            seg = seg + AudioSegment.silent(CHUNK_GAP_MS)
            fname = f"{seq:04d}.wav"
            out_path = os.path.join(OUT_DIR, fname)
            seg.export(out_path, format="wav")
            mf_writer.writerow({"seq":seq, "filename":out_path, "type":"narrator", "char_id":"", "char_name":"narrator", "gender":"female", "gender_reason":"narrator_default", "text":c})
            mf_handle.flush()
            try: os.fsync(mf_handle.fileno())
            except Exception: pass
            print("WROTE:", out_path)
            seq += 1

    mf_handle.close()
    print("Done. Generated", seq-1, "WAVs in", OUT_DIR)
    print("Manifest at", MANIFEST_CSV)
    print("Gender report at", GENDER_REPORT_JSON)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal:", e)
        traceback.print_exc()
        sys.exit(1)
