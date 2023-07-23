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

st.subheader("Please select the page you want to explore: ")
url_1= "http://localhost:8501/Explore"
url_2 = "http://localhost:8501/MiniLM"
url_3= "http://localhost:8501/DistilledBert"


if st.button("Explore"):
    webbrowser.open_new_tab(url_1)

if st.button("MiniML"):
    webbrowser.open_new_tab(url_2)

if st.button("RoBerta"):
    webbrowser.open_new_tab(url_3)
