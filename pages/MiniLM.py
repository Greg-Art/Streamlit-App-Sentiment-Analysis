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

path= "./models/MiniLM"
##tokenizer= AutoTokenizer.from_pretrained("gArthur98/Greg-Sentiment-classifier")

##model= AutoModel.from_pretrained("gArthur98/Greg-Sentiment-classifier")

##model.save_pretrained(path)
##tokenizer.save_pretrained(path)

model= AutoModel.from_pretrained(path)
tokenizer= AutoTokenizer.from_pretrained(path)


st.title("Welcome to the MiniLM Model Page")

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

pipe= pipeline("text-classification", model=path, tokenizer=path)

result= pipe(input)

final = st.button("Calculate Sentiment")

if final:
    for results in result:
        if results['label']== 'LABEL_0':
            st.write(f"Your sentiment is Negative with a confidence score of {results['score']}")
        elif results["label"]=='LABEL_1':
           st.write(f"Your sentiment is Neutral with a confidence score of {results['score']}")
        else:
           st.write(f"Your sentiment is Positive with a confidence score of {results['score']}")



