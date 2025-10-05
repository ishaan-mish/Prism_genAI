README — Audio generation from BookNLP outputs (beginner friendly)

This guide shows step-by-step how to:

Create a simple folder structure for your book and outputs.

Install BookNLP and required models.

Patch a small BookNLP encoding bug (Windows).

Prepare BookNLP model files.

Run the provided Bark-based audio generator script to turn book quotes into WAV files.

Inspect the generated WAV files and manifest.

Everything here is beginner-friendly and written so you can copy/paste commands.

Create project folders

Create a folder to hold everything. Example path:
C:\Users\golec\Downloads\final fantasy

Inside that folder create two subfolders:

INPUT (put your source novel here as a single text file)

VOICOVERS (the script will read BookNLP outputs placed here and write WAVs here)

Example structure:
C:\Users\golec\Downloads\final fantasy
├─ INPUT
│ └─ NOVEL.txt <- save your book here as plain UTF-8 text
└─ VOICOVERS\ <- BookNLP outputs + generated WAVs will go here

After you save your book in INPUT, right-click the file and choose "Copy as path"
so you have the full path for later steps.

Create conda environment and install BookNLP

Open Anaconda Prompt and run:

(Replace golec with your username and use the folder you created.)

cd C:\Users\golec\Downloads\final fantasy

conda create --name booknlp python=3.10 -y
conda activate booknlp

pip install booknlp
python -m spacy download en_core_web_sm


Notes:

BookNLP installation creates a booknlp_models folder under your user directory (see step 4).

If pip install booknlp fails, make sure your internet is available and try again.

Prepare booknlp model folders (download BERT models)

Create a folder C:\Users\golec\booknlps (replace golec with your username).

Inside it create these three empty subfolders:

coref_google

entities_google

speaker_google

Open each folder in a terminal (right-click inside the folder → "Open in Terminal") and run the appropriate git clone commands to download the pretrained model files:

In coref_google folder:

git clone https://huggingface.co/google/bert_uncased_L-12_H-768_A-12


In entities_google folder:

git clone https://huggingface.co/google/bert_uncased_L-6_H-768_A-12


In speaker_google folder:

git clone https://huggingface.co/google/bert_uncased_L-12_H-768_A-12


Wait for each clone to finish before closing that terminal.

Patch BookNLP encoding bug (Windows)

On some Windows installs BookNLP can fail reading hyperparam files due to encoding. Edit this file:

C:\Users\golec\anaconda3\Lib\site-packages\booknlp\english\gender_inference_model_1.py


(Replace golec with your username.)

Open it in a text editor and find the function:

def read_hyperparams(self, filename):
    self.hyperparameters={}
    with open(filename) as file:
        header=file.readline().rstrip()
        ...


Change with open(filename) as file: to:

with open(filename, encoding='UTF8') as file:


Save the file. This fixes Windows encoding problems.

Run BookNLP to produce the five files

A simple runner script (from the BookNLP repo) expects paths to:

input file (your NOVEL.txt)

output folder

model files

Example variables you must set in runner.py (open the runner file and edit these lines):

input_file = Path(r"C:\Users\golec\Downloads\final fantasy\INPUT\NOVEL.txt")
output_directory = Path(r"C:\Users\golec\Downloads\final fantasy\VOICOVERS")
raw_book_id = "a study in scarlet (actuak 8npuys)"


Next, specify model locations (these files are created under your user booknlp_models folder by pip install; copy actual .model paths and paste here):

coref_path  = Path(r"C:\Users\golec\booknlp_models\coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model")
entity_path = Path(r"C:\Users\golec\booknlp_models\entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model")
quote_path  = Path(r"C:\Users\golec\booknlp_models\speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model")


Run BookNLP runner:

python runner.py


After BookNLP runs, the VOICOVERS folder should now contain these files (names similar to these):

a_study_in_scarlet_actuak_8npuys.quotes

a_study_in_scarlet_actuak_8npuys.entities

full_text.txt (or similarly named file) — this is crucial: offsets must match full text

plus other supporting token files

Install Bark & audio dependencies

Open the same conda environment or create a separate one for Bark (example used earlier: vn_bark). From Anaconda Prompt run:

conda create --name vn_bark python=3.10 -y
conda activate vn_bark

pip install bark pydub numpy scipy
# install torch appropriate for your machine - see pytorch.org for correct command
pip install torch
# optional but helpful:
pip install gender-guesser


Also install ffmpeg on your system and add ffmpeg.exe to PATH — pydub needs it.

Place the Bark audio generator script

Put the script generate_bark_audio_from_booknlp_final.py in the C:\Users\golec\Downloads\final fantasy folder (or the project folder you created).

Top-of-script variables to update (if different):

BASE_DIR = r"C:\Users\golec\Downloads\final fantasy\VOICOVERS"
QUOTES_FILE = os.path.join(BASE_DIR, "a_study_in_scarlet_actuak_8npuys.quotes")
ENTITIES_FILE = os.path.join(BASE_DIR, "a_study_in_scarlet_actuak_8npuys.entities")
FULL_TEXT_FILE = os.path.join(BASE_DIR, "full_text.txt")
OUT_DIR = os.path.join(BASE_DIR, "wavs")
MANIFEST_CSV = os.path.join(BASE_DIR, "manifest.csv")


Make sure those filenames match what BookNLP produced.

Run the Bark audio generator

Activate the bark environment and run:

conda activate vn_bark
python generate_bark_audio_from_booknlp_final.py


What happens:

The script writes gender_report.json first to help you check resolved genders.

Then it generates WAV files into OUT_DIR (default: .../VOICOVERS/wavs).

By default the script writes manifest.csv at the end of the run, containing:
seq, filename, type, char_id, char_name, text

WAV files are named 0001.wav, 0002.wav, etc.

Check progress while generation runs (optional but recommended)

By default manifest.csv is written when the script finishes. If you want the manifest and console output during the run, apply this tiny patch to the script.

A) At the top of main() (after you set MANIFEST_CSV) add these lines:

mf_handle = open(MANIFEST_CSV, "w", newline="", encoding="utf-8")
mf_writer = csv.DictWriter(mf_handle, fieldnames=["seq","filename","type","char_id","char_name","text"])
mf_writer.writeheader()


B) Replace the place where the script exports WAV and appends to manifest with this pattern (inside the loop):

out_path = os.path.join(OUT_DIR, fname)
seg.export(out_path, format="wav")
print("WROTE:", out_path, flush=True)

# append to manifest on disk
mf_writer.writerow({"seq":seq, "filename":out_path, "type":"quote", "char_id":cid, "char_name":char_name, "text":c})
mf_handle.flush()
os.fsync(mf_handle.fileno())
seq += 1


C) At the very end of main() before exiting add:

mf_handle.close()


This change writes each manifest row as soon as a WAV is created, and prints filenames to the console.

Inspect the generated files

WAVs: open folder C:\Users\golec\Downloads\final fantasy\VOICOVERS\wavs

Manifest (incremental or final): C:\Users\golec\Downloads\final fantasy\VOICOVERS\manifest.csv

Gender audit: C:\Users\golec\Downloads\final fantasy\VOICOVERS\gender_report.json

Count WAV files (PowerShell):

(Get-ChildItem "C:\Users\golec\Downloads\final fantasy\VOICOVERS\wavs" -Filter *.wav).Count


Edit gender overrides (if voices are wrong)

Open the Bark script and edit GENDER_OVERRIDES near the top. Acceptable keys:

integer cluster id (e.g., 20: "male")

exact name string (case-insensitive) (e.g., "Sherlock Holmes": "male")

regex prefix re: (e.g., "re:^dr\\s+watson": "male")

After editing, re-run the script. The script writes gender_report.json at the start so you can see what it chose and why.

Troubleshooting tips

If you see No GPU being used. Careful, inference might be very slow! — it means Bark will run on CPU; expect much slower generation.

If Bark fails the first time, ensure it can download models (run with internet connected).

If you see RequestsDependencyWarning install chardet:

pip install chardet


If no WAVs appear: check that quote_start and quote_end offsets align with full_text.txt. Offsets must be correct.

If manifest.csv isn't visible mid-run, apply the incremental manifest patch (step 9).

Extra small utilities (optional)

A) To create an editable CSV of resolved genders (so you can load in Excel and edit), add this snippet after gender_report.json is written:

import csv
with open(os.path.join(BASE_DIR, "gender_report.csv"), "w", newline="", encoding="utf-8") as gf:
    w = csv.writer(gf)
    w.writerow(["cluster_id","gender","reason"])
    for k,v in report.items():
        w.writerow([k, v["gender"], v["reason"]])


B) To run the generator with a different output folder from the command line, add argparse support at top:

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, help="output directory for wavs", default=None)
args = parser.parse_args()
if args.outdir:
    OUT_DIR = args.outdir
    os.makedirs(OUT_DIR, exist_ok=True)


Then run:

python generate_bark_audio_from_booknlp_final.py --outdir "D:\audio_output"


Quick checklist before running

 NOVEL.txt saved in INPUT folder (UTF-8)

 BookNLP runner configured with correct model paths

 BookNLP completed and .quotes, .entities, full_text.txt exist in VOICOVERS

 Bark environment created and libraries installed

 generate_bark_audio_from_booknlp_final.py placed in base folder and paths inside updated

 (Optional) incremental manifest patch applied if you want live updates

 Run the script and check wavs folder and manifest.csv