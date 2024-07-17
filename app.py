import streamlit as st

from models.chat_gpt.chat_gpt import OpenAIGPT
from models.stable_diffusion_xl_base_1.stable_diffusion_xl_base_1 import StableDiffusionXLBase1

openai_gpt = OpenAIGPT(api_key="")

st.set_page_config(
    page_title="Text to Image Page",
    page_icon="🧊",
    layout="centered",
)
st.title("Tien's Text to Image Page")
st.subheader("Use multiple models to generate images from text")

prompt_text = st.text_area("Enter your prompt", "A beautiful sunset over the city")
# streamlit space
st.text(" ")
numbers_to_generate = st.number_input("How many images do you wish to generate?", 1, 10, 1)
st.text(" ")
st.text("Note:\n"
        "1. openai [0.08$ for 1 request](Number of images * 0.08$ will be charged)\n"
        "2. stable_diffusion_xl_base_1.0 "
        "[free but time-consuming](1->5p in MAC M3 PRO because Chip AppleSilicon is not have CUDA)")
all_models = [
    "openai",
    "stable_diffusion_xl_base_1.0"
]
options = st.multiselect(
    "What are the models you want to use?",
    all_models,
    all_models[0],
)
st.write("You selected:", options)
st.text(" ")
if st.button("Generate Images"):

    container = st.container(border=True)
    container.write("Generating images...")
    container.write(f"Your prompt: {prompt_text}")
    container.write(f"Number of images to generate: {numbers_to_generate}")
    container.write(f"Models selected: {options}")

    map_model_to_images = {}
    for model in all_models:
        if model in options:
            map_model_to_images[model] = []
            if model == all_models[0]:
                print("Generating images using openai (chat_gpt)...")
                images = openai_gpt.text_to_image(prompt_text, numbers_to_generate)
                map_model_to_images[model].append(images)
            elif model == all_models[1]:
                print("Generating images using stable_diffusion_xl_base_1.0 (huggingface)...")
                stable_diffusion_xl_base_1 = StableDiffusionXLBase1()
                images = stable_diffusion_xl_base_1.text_to_image(prompt_text, numbers_to_generate)
                map_model_to_images[model].append(images)
            else:
                map_model_to_images[model].append([])

    for model, images in map_model_to_images.items():
        st.subheader(f"Images from {model}")
        for image in images:
            st.image(image, use_column_width=True)
