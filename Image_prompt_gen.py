import google.generativeai as genai

# ✅ Your API Key from Google AI Studio
genai.configure(api_key="AIzaSyAr39ItVMv2J08iLT5hG9fpoUZlJBngGCA")

# Use the short model name (AI Studio format)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

def make_image_prompt(paragraph: str) -> str:
    instruction = (
        "You are a creative assistant that converts book paragraphs into "
        "detailed visual descriptions for illustrations. Focus on capturing "
        "the setting, characters, emotions, and atmosphere in a way that an "
        "AI image generator can easily understand. The output should be one "
        "or two concise yet vivid descriptions that emphasize imagery, not "
        "abstract interpretation.\n\n"
        f"Paragraph: {paragraph}\n\n"
        "Illustration Prompt:"
    )

    response = model.generate_content(instruction)
    return response.text.strip()

if __name__ == "__main__":
    paragraph = input("📖 Enter a paragraph from your book:\n> ")
    print("\n✨ Generated Image Prompt:")
    print(make_image_prompt(paragraph))
