# runner_fixed_full.py
import time
import sys
import traceback
import tempfile
import re
from pathlib import Path
import builtins

import torch
from booknlp.booknlp import BookNLP

# === Your original model file paths ===
coref_path  = Path(r"C:\Users\golec\booknlp_models\coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model")
entity_path = Path(r"C:\Users\golec\booknlp_models\entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model")
quote_path  = Path(r"C:\Users\golec\booknlp_models\speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model")

# === Your actual input / output settings ===
input_file = Path(r"C:\Users\golec\Downloads\final fantasy\INPUT\NOVEL.txt")
output_directory = Path(r"C:\Users\golec\Downloads\final fantasy\OUTPUT")
raw_book_id = "a study in scarlet (actuak 8npuys)"  

def safe_book_id(s: str) -> str:
    # convert to lowercase, replace non-alnum with underscore, collapse underscores
    s2 = s.lower()
    s2 = re.sub(r'[^a-z0-9]+', '_', s2).strip('_')
    return s2 or "book"

def clean_checkpoint(in_path: Path) -> Path:
    """Create a cleaned copy of in_path with '_modified' before the suffix.
       If cleaned file already exists, return it.
    """
    if not in_path.exists():
        raise FileNotFoundError(f"Model file not found: {in_path}")

    out_path = in_path.with_name(in_path.stem + "_modified" + in_path.suffix)
    if out_path.exists():
        print(f"[skip] cleaned file already exists: {out_path}")
        return out_path

    print(f"Loading checkpoint: {in_path}")
    state = torch.load(str(in_path), map_location="cpu")

    # handle wrapped checkpoint with 'state_dict'
    if isinstance(state, dict) and "state_dict" in state:
        inner = state["state_dict"]
        wrapper = True
    else:
        inner = state
        wrapper = False

    if isinstance(inner, dict):
        removed = [k for k in inner.keys() if "position_ids" in k]
        if removed:
            print(f"  removing keys: {removed}")
        else:
            print("  no 'position_ids' keys found.")
        cleaned = {k: v for k, v in inner.items() if "position_ids" not in k}
    else:
        print("  checkpoint inner is not a dict; saving as-is")
        cleaned = inner

    if wrapper:
        state["state_dict"] = cleaned
        torch.save(state, str(out_path))
    else:
        torch.save(cleaned, str(out_path))

    print(f"Saved cleaned model to: {out_path}")
    return out_path

def make_utf8_copy(src_path: Path) -> Path:
    """Try some encodings and write a UTF-8 temporary copy. Returns the temp path."""
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1"]
    data = None
    for enc in encodings_to_try:
        try:
            with src_path.open("r", encoding=enc) as f:
                data = f.read()
            print(f"[ok] read {src_path} using encoding: {enc}")
            break
        except UnicodeDecodeError:
            print(f"[fail] decoding with {enc}")
        except Exception as e:
            print(f"[fail] reading file with {enc}: {e}")

    if data is None:
        print("Falling back to bytes read + 'replace' decoding (may lose some chars).")
        raw = src_path.read_bytes()
        data = raw.decode("utf-8", errors="replace")

    tmp_dir = Path(tempfile.gettempdir())
    tmp_file = tmp_dir / (src_path.stem + "_utf8_tmp" + src_path.suffix)
    with tmp_file.open("w", encoding="utf-8") as out:
        out.write(data)
    print(f"Wrote UTF-8 copy to: {tmp_file}")
    return tmp_file

def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def monkeypatch_open_to_utf8():
    """Monkeypatch builtins.open so text opens default to utf-8 when encoding not provided.
       Returns the original open so caller can restore it.
    """
    _orig_open = builtins.open

    def _utf8_open(file, mode='r', *args, **kwargs):
        # Only set encoding for text-mode opens and when no encoding was explicitly provided
        if 'b' not in mode and 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'
        return _orig_open(file, mode, *args, **kwargs)

    builtins.open = _utf8_open
    return _orig_open

def restore_open(orig_open):
    builtins.open = orig_open

def main():
    start = time.time()
    utf8_input = None
    orig_open = None
    try:
        # Verify input exists
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # 1) Clean model files (create _modified copies)
        coref_clean = clean_checkpoint(coref_path)
        entity_clean = clean_checkpoint(entity_path)
        quote_clean = clean_checkpoint(quote_path)

        # 2) Make UTF-8 copy of input (safer for BookNLP processing)
        utf8_input = make_utf8_copy(input_file)

        # 3) Ensure output directory exists
        ensure_output_dir(output_directory)

        # 4) Build model params using cleaned models
        model_params = {
            "pipeline": "entity,quote,supersense,event,coref",
            "model": "custom",
            "entity_model_path": str(entity_clean),
            "coref_model_path": str(coref_clean),
            "quote_attribution_model_path": str(quote_clean),
            "bert_model_path": rf"C:\Users\golec\.cache\huggingface\hub"
        }

        print("\nInitializing BookNLP with custom cleaned models...")
        print(model_params)
        booknlp = BookNLP("en", model_params)
        print("--- startup: {:.3f} seconds ---".format(time.time() - start))

        # 5) Monkeypatch open -> utf-8 and run processing
        orig_open = monkeypatch_open_to_utf8()
        try:
            safe_id = safe_book_id(raw_book_id)
            print(f"Processing book id: {safe_id}")
            booknlp.process(str(utf8_input), str(output_directory), safe_id)
            print("Processing finished successfully.")
        finally:
            # restore builtins.open
            if orig_open:
                restore_open(orig_open)

        print(f"Elapsed: {time.time() - start:.1f}s")

    except Exception:
        print("Error during cleaning or BookNLP run — full traceback follows:")
        traceback.print_exc()
        sys.exit(1)

    finally:
        # cleanup temporary UTF-8 copy if present
        try:
            if utf8_input and utf8_input.exists():
                utf8_input.unlink()
                print(f"Removed temporary file: {utf8_input}")
        except Exception:
            pass

if __name__ == "__main__":
    main()
