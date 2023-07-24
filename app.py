import streamlit as st
import plotly as plt 
import numpy as np 
import pandas as pd
import webbrowser

st.set_page_config(page_title= "Welcome Page", page_icon ="👋")


st.sidebar.success("select a model to use")

st.title("Welcome to my Sentiment Analysis App")

st.markdown("""This project was done by Group Seville and it encapsulates two sentiment analysis models: \n
         
- Roberta Base 

- MiniLM
            
In this project, you can be able to predict your sentiments with these models. 
         
         """)

st.write("""

We trained the models on a dataset which contain dirty data which we run

""")
