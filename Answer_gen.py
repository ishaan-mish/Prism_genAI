import google.generativeai as genai

# ✅ Your API Key from Google AI Studio
genai.configure(api_key="AIzaSyAr39ItVMv2J08iLT5hG9fpoUZlJBngGCA")

# Use the short model name (AI Studio format)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

def generate_answer(text: str, question: str) -> str:
    """
    Generate a detailed, creative answer to a question based on a provided text.
    The model is instructed to cover multiple aspects: explanation, interpretation,
    real-world connections, alternate timelines, and deep exploration of the scene or concepts.
    """
    instruction = (
        "You are an expert assistant that provides highly detailed, thoughtful, "
        "and creative answers based on a given text. The user may ask any type of question: "
        "explaining meaning, connecting it to real-world situations, imagining alternate timelines, "
        "analyzing characters, exploring emotions, or exploring hidden details. "
        "Your response should be clear, insightful, and engaging, covering multiple perspectives "
        "if relevant. Be thorough yet concise, and always base your answer on the provided text.\n\n"
        f"Text:\n{text}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    response = model.generate_content(instruction)
    return response.text.strip()


if __name__ == "__main__":
    # Upload text or input it directly
    text = input("📖 Enter the text to reference:\n> ")
    
    while True:
        question = input("\n❓ Ask any question about the text (or type 'exit' to quit):\n> ")
        if question.lower() in ["exit", "quit"]:
            break
        answer = generate_answer(text, question)
        print("\n✨ Generated Answer:")
        print(answer)
