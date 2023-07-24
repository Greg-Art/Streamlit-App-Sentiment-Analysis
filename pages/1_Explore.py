import streamlit as st 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as pe
from wordcloud import WordCloud, STOPWORDS
from nltk import FreqDist

st.set_option('deprecation.showPyplotGlobalUse', False)



st.title("Welcome! On This Page, you will be able to see the various visuals used for our EDA")

##loading my dataset 

data= pd.read_csv("datasets/Train.csv")

clean_data= pd.read_csv("datasets/clean_copy.csv")

clean_data= clean_data.dropna()

##plotting my wordcloud for the unclean dataset

unclean_words= " ".join(data["safe_text"])

wc= WordCloud(stopwords=STOPWORDS).generate(unclean_words)

plt.figure(figsize= (5,10))
plt.title("Most common Words in unclean Dataset")
plt.imshow(wc)
st.pyplot()

##creating a wordcloud of my most common word in cleaned tweet
clean_words= ' '.join(clean_data["clean_tweet"]).split() ##converting the dataframe to corpus of words

freq_words= pd.DataFrame(FreqDist(clean_words).most_common(20), columns= ["word", "count"])

fig= pe.treemap(data_frame=freq_words, path=["word"], values= "count", title= "Top 20 Most Frequent Words After Cleaning")

st.plotly_chart(fig)


##getting the tweet lengths
data["tweet_length"]= [len(i.split(" ")) for i in data["safe_text"]]

words= data["tweet_length"].value_counts().reset_index()
words
fig_2= pe.scatter(data_frame=words, x="tweet_length", y="count", size= "count")

st.plotly_chart(fig_2)