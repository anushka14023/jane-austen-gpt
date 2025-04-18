import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Streamlit UI Setup
st.set_page_config(page_title="Austen Style Generator", page_icon="📚", layout="centered")
st.title("📖 Jane Austen Style Text Generator")
st.markdown("Enter a prompt and generate elegant prose in Jane Austen’s signature style!")

# Load fine-tuned model and tokenizer (replace with your model path if hosted locally or on Hugging Face)
@st.cache_resource
def load_model():
    # If hosted on Hugging Face
    model_name = "your-username/austen-style-gpt2"  # Replace with your fine-tuned model repo name
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Text input
prompt = st.text_area("Enter your prompt:", "It is a truth universally acknowledged")

# Generate text when button clicked
if st.button("Generate Text"):
    with st.spinner("Generating..."):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=200,
                num_return_sequences=1,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=2,
                do_sample=True,
                early_stopping=True
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        st.subheader("💬 Generated Text")
        st.write(generated_text)
