import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Set page config
st.set_page_config(page_title="Austen Style Generator", page_icon="📚", layout="centered")

# Title
st.title("📖 Jane Austen Style Text Generator")
st.markdown("This app generates elegant prose using the GPT-2 language model. Start with a prompt!")

# Load GPT-2 model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Prompt input
prompt = st.text_area("Enter your prompt:", "It is a truth universally acknowledged")

# Generate text
if st.button("Generate"):
    with st.spinner("Generating... please wait"):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=200,
                num_return_sequences=1,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=2,
                do_sample=True,
                early_stopping=True
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        st.subheader("💬 Generated Text")
        st.write(generated_text)
