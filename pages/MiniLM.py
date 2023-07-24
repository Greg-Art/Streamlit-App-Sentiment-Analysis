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


st.title("Welcome to the MiniLM Model Page")


@st.cache_data
def load_model(model_name):
    model= AutoModelForSequenceClassification.from_pretrained(model_name)
    return model

##creating my model
ml_model= load_model("gArthur98/Greg-Sentiment-classifier")

@st.cache_data
def load_tokenizer(tokenizer_name):
    tokenizer= AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer

ml_token= load_tokenizer("gArthur98/Greg-Sentiment-classifier") 

##front end 

text= st.text_input("Please Enter Your Sentence Below: ")

##cleaning

def data_cleaner(text):
    text= text.lower()
     ##removing has tags
    text= re.sub(r'#\w+', '', text)
  ##removing punctuations
    text= re.sub("[^\w\s]", repl= "", string=text)
    text= re.sub(r'\d+', '', text)
    text= " ".join([word for word in text.split() if not word.isdigit()]) ##removing digits
    return text

##running my input through my function

input_text= data_cleaner(text)

pipe= pipeline("text-classification", model=ml_model, tokenizer=ml_token)

result= pipe(input_text)

final = st.button("Calculate Sentiment")

if final:
    for results in result:
        if results['label']== 'LABEL_0':
            st.write(f"Your sentiment is Negative with a confidence score of {results['score']}")
        elif results["label"]=='LABEL_1':
           st.write(f"Your sentiment is Neutral with a confidence score of {results['score']}")
        else:
           st.write(f"Your sentiment is Positive with a confidence score of {results['score']}")
