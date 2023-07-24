import streamlit as st
import plotly as plt 
import numpy as np 
import pandas as pd
import webbrowser

st.set_page_config(page_title= "Welcome Page", page_icon ="👋")


st.sidebar.success("select a model to use")

st.title("Welcome to my Sentiment Analysis App")

st.markdown("""This project was done by Gregory Arthur, a budding data scientist from Ghana. On this page, you will 
be able to classify Covid-19 sentiments with the Roberta Base model

             
         """)

st.write("""

We trained the models on a dataset which contain dirty data which we run

""")
