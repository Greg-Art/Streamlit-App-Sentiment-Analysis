import streamlit as st 
import torch 
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
import numpy as np
import pandas as pd
import re
from scipy.special import softmax
from transformers import pipeline
import xformers

## Creating a cache to store my model for efficiency
@st.cache_data(ttl=86400)
def load_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model

## Creating my tokenizer
@st.cache_data(ttl=86400)
def load_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer

## Front end
st.title("Welcome to the MiniLM Model Page")

text = st.text_input("Please Enter Your Sentence Below: ")

## Cleaning
def data_cleaner(text):
    text = text.lower()
    ## Removing hashtags
    text = re.sub(r'#\w+', '', text)
    ## Removing punctuations
    text = re.sub("[^\w\s]", repl="", string=text)
    text = re.sub(r'\d+', '', text)
    text = " ".join([word for word in text.split() if not word.isdigit()])
    return text

## Running my input through my function
input_text = data_cleaner(text)

if 'ml_model' not in st.session_state:
    st.session_state.ml_model = load_model("gArthur98/Greg-Sentiment-classifier")

if 'ml_token' not in st.session_state:
    st.session_state.ml_token = load_tokenizer("gArthur98/Greg-Sentiment-classifier")

pipe = pipeline("text-classification", model=st.session_state.ml_model, tokenizer=st.session_state.ml_token)

result = pipe(input_text)

final = st.button("Calculate Sentiment")

if final:
    for results in result:
        if results['label'] == 'LABEL_0':
            st.write(f"Your sentiment is Negative with a confidence score of {results['score']}")
        elif results["label"] == 'LABEL_1':
           st.write(f"Your sentiment is Neutral with a confidence score of {results['score']}")
        else:
           st.write(f"Your sentiment is Positive with a confidence score of {results['score']}")

