
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
import requests
import json

from streamlit_lottie import st_lottie 


import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from transformers import pipeline

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
st.title("Welcome to the Fine-Tuned RoBerta Sentiment Classification Model Page")

##including an animation to my page 

@st.cache_data ##adding a cache
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
   

lottie_hello= load_lottiefile("./lottie_animations/roberta.json") ##uploading my gif

st_lottie(lottie_hello, height= 200) ##loading my gif

text = st.text_input("Please Enter a Covid-19 Themed Sentence Below: ")

st.write("""Example of sentences /n
         
         - I hate the vaccine /n
         - I love the vaccine /n
         - Covid-19 is Moving Fast
    
         """)

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
text_input = data_cleaner(text)

if 'ro_model' not in st.session_state:
    st.session_state.ro_model = load_model("gArthur98/Roberta-classweight-Sentiment-classifier")

if 'ro_token' not in st.session_state:
    st.session_state.ro_token = load_tokenizer("gArthur98/Roberta-classweight-Sentiment-classifier")

pipe = pipeline("sentiment-analysis", model=st.session_state.ro_model, tokenizer=st.session_state.ro_token)

result = pipe(text_input)

final = st.button("Predict Sentiment")

## Initializing my session state
if final:
    for results in result:
        if results['label'] == 'LABEL_0':
            st.write(f"Your sentiment is Negative with a confidence score of {results['score']}")
        elif results["label"] == 'LABEL_1':
           st.write(f"Your sentiment is Neutral with a confidence score of {results['score']}")
        else:
           st.write(f"Your sentiment is Positive with a confidence score of {results['score']}")



