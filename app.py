import streamlit as st
import plotly as plt 
import numpy as np 
import pandas as pd
import webbrowser

st.set_page_config(page_title= "Welcome Page", page_icon ="👋")


st.sidebar.success("select a model to use")

st.title("Welcome to my Sentiment Analysis App")

##including an animation to my page

@st.cache_data ##adding a cache
def load_lottiefile(url: str):
    r= requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

##downloading  the animation

lottie_hello= load_lottiefile("https://lottiefiles.com/animations/face-animation-qDPFBv1QlA")


st.markdown("""On this app, you will  be able to classify Covid-19 sentiments with the Roberta Base model
The objective of this challenge is to develop a machine learning model to assess if a twitter post that is related to vaccinations is positive, neutral, or negative.""")

st.subheader("""Variable Definition:""")

st.write("""

    **tweet_id**: Unique identifier of the tweet

**safe_tweet**: Text contained in the tweet. Some sensitive information has been removed like usernames and urls

**label**: Sentiment of the tweet (-1 for negative, 0 for neutral, 1 for positive)

**agreement**: The tweets were labeled by three people. Agreement indicates the percentage of the three reviewers that agreed on the given label. You may use this column in your training, but agreement data will not be shared for the test set.

**Train.csv**:  Labelled tweets on which to train your model
             
         """)

data= pd.read_csv("datasets/Train.csv")

st.subheader("A sample of the orginal Dataframe (Train.csv)")

st.write(data.head())

st.subheader("A sample of the preprocessed dataset")

data_clean= pd.read_csv("datasets/clean_copy.csv")

data_clean= data_clean.drop("Unnamed: 0", axis= 1)

st.write(data_clean.head())