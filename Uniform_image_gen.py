import streamlit as st
import os
import requests
from dotenv import load_dotenv

# -----------------------
# Load Hugging Face Token
# -----------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    st.error("HF_TOKEN not loaded! Check your .env file.")
    st.stop()

# -----------------------
# Model configuration
# -----------------------
# Replace MODEL_ID with any free Hugging Face hosted model
MODEL_ID = "black-forest-labs/FLUX.1-dev"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# -----------------------
# Style instructions
# -----------------------
STYLE_PROMPT = (
    "in the style of a detailed digital painting, "
    "cinematic lighting, soft pastel colors, "
    "consistent character design, smooth textures, "
    "highly detailed environment, cohesive composition"
)

NEGATIVE_PROMPT = "lowres, bad anatomy, deformed, text, watermark"

# -----------------------
# Streamlit Page Config
# -----------------------
st.set_page_config(
    page_title="Samsung Prism Project",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("Samsung Prism Project")
st.sidebar.info(
    """
    Welcome to the **Samsung Prism Project** 🎨  
    Generate AI images with a **consistent style** using a free Hugging Face model.
    """
)
st.sidebar.markdown("---")
st.sidebar.write("**Instructions:**")
st.sidebar.write("1. Enter your main prompt in the text box below.")
st.sidebar.write("2. (Optional) Add extra style instructions in the optional section.")
st.sidebar.write("3. Click 'Generate Image' and download your image!")

# -----------------------
# Main App UI
# -----------------------
st.title("🎨 Samsung Prism Project")
st.markdown(
    """
    Generate AI images in a **consistent art style** using a free Hugging Face model.  
    Enter your creative prompt below and click **Generate Image**.
    """
)

# Main prompt
prompt = st.text_area("Enter your main prompt here:", height=150)

# Optional extra style
st.markdown("### Optional: Additional Style Instructions")
optional_style = st.text_area(
    "Add extra style keywords or instructions (optional):", height=100
)

# Sliders for generation parameters
st.markdown("### Generation Settings")
num_steps = st.slider("Number of Inference Steps:", min_value=10, max_value=100, value=50, step=5)
guidance_scale = st.slider("Guidance Scale:", min_value=1.0, max_value=15.0, value=7.5, step=0.5)

# Generate Image Button
if st.button("🚀 Generate Image"):
    if not prompt.strip():
        st.error("Please enter a main prompt to generate an image!")
    else:
        # Combine main prompt + fixed style + optional style
        final_prompt = prompt + ", " + STYLE_PROMPT
        if optional_style.strip():
            final_prompt += ", " + optional_style

        payload = {
            "inputs": final_prompt,
            "parameters": {
                "num_inference_steps": num_steps,
                "guidance_scale": guidance_scale,
                "negative_prompt": NEGATIVE_PROMPT
            }
        }

        with st.spinner("Generating image... ⏳"):
            response = requests.post(API_URL, headers=HEADERS, json=payload)
            if response.status_code == 200:
                img_bytes = response.content
                st.image(img_bytes, caption=f"Prompt: {final_prompt}", use_column_width=True)
                st.download_button("⬇️ Download Image", img_bytes, file_name="generated_image.png")
            else:
                st.error(f"❌ Error: {response.status_code}\n{response.text}")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("Made with ❤️ by **Samsung Prism Team**")
