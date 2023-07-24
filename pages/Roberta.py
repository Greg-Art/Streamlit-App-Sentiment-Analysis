import streamlit as st 

st.title("Welcome to the RoBerta Model Page")

import streamlit as st 
import torch 
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoConfig
import numpy as np
import pandas as pd
import re
from scipy.special import softmax
from transformers import pipeline

##creating a directory

##directory= "./models/Roberta"
##model= AutoModel.from_pretrained("gArthur98/Roberta-Sentiment-classifier")
##tokenizer= AutoTokenizer.from_pretrained("gArthur98/Roberta-Sentiment-classifier")

##model.save_pretrained(directory)
##tokenizer.save_pretrained(directory)

##tokenizer= AutoTokenizer.from_pretrained(directory)
##model= AutoModel.from_pretrained(directory)

##@st.cache_data
##def load_model(model_name):
    ##model= AutoModel.from_pretrained(model_name)
    ##return model

##@st.cache_data
##def load_tokenizer(tokenizer_name):
   ## tokenizer= AutoTokenizer.from_pretrained(tokenizer_name)
    ##return tokenizer

##model= load_model("gArthur98/Roberta-Sentiment-classifier")

##tokenizer= load_tokenizer("gArthur98/Roberta-Sentiment-classifier")
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
    text= " ".join([word for word in text.split() if not word.isdigit()])
    return text

##running my input through my function

input= data_cleaner(text)

pipe= pipeline("sentiment-analysis", model="gArthur98/Roberta-Sentiment-classifier", tokenizer="gArthur98/Roberta-Sentiment-classifier")

result= pipe(input)

final = st.button("Predict Sentiment")

if final:
    for results in result:
        if results['label']== 'LABEL_0':
            st.write(f"Your sentiment is Negative with a confidence score of {results['score']}")
        elif results["label"]=='LABEL_1':
           st.write(f"Your sentiment is Neutral with a confidence score of {results['score']}")
        else:
           st.write(f"Your sentiment is Positive with a confidence score of {results['score']}")