import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# App UI
st.title("📖 Jane Austen Text Generator")
st.markdown("Generate text in the style of Jane Austen using GPT-2.")

# Input prompt
prompt = st.text_area("Enter your prompt:", "It is a truth universally acknowledged")

if st.button("Generate"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating..."):
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            outputs = model.generate(
                inputs,
                max_length=150,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.markdown("### ✍️ Generated Text")
            st.write(generated_text)
