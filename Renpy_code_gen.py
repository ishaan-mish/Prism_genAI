import google.generativeai as genai
import csv
import os

# ✅ API Key
genai.configure(api_key="AIzaSyAr39ItVMv2J08iLT5hG9fpoUZlJBngGCA")
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Load voices from CSV
def load_character_audio(csv_file="characters.csv"):
    voices = {}
    if os.path.exists(csv_file):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) == 2:
                    voices[row[0].strip()] = row[1].strip()
    return voices


def make_rpy_script(paragraph: str, character_audio: dict) -> str:
    instruction = (
        "You are an assistant that converts story paragraphs into Ren'Py (.rpy) script.\n"
        "⚠️ IMPORTANT RULES:\n"
        "1. Output ONLY valid Ren'Py code — no explanations, no markdown, no comments.\n"
        "2. Assume character images are named like `harry_potter.png` and can be shown with states (e.g., worried, happy).\n"
        "3. Use audio for dialogue and narration from the mapping below.\n"
        "4. Use `scene_1.mp3`, `scene_2.mp3` etc. as background narration when appropriate.\n"
        "5. Incorporate character positions (left, right, center) and emotional expressions if possible.\n"
        "6. Treat the input paragraph as a scene to be fully scripted.\n\n"
        f"Character Audio Mapping: {character_audio}\n\n"
        f"Story Paragraph:\n{paragraph}\n\n"
        "Now generate the COMPLETE Ren'Py code for this scene."
    )

    response = model.generate_content(instruction)
    return response.text.strip()


def save_rpy(script: str, filename="current.rpy"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"✅ Ren'Py script saved as {filename}")


if __name__ == "__main__":
    paragraph = input("📖 Enter a paragraph from your story:\n> ")
    char_audio = load_character_audio("characters.csv")

    script = make_rpy_script(paragraph, char_audio)
    save_rpy(script)

    print("\n✨ Generated Ren'Py Script:\n")
    print(script)
