# importing the libraries
import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import pandas as pd
import contractions
import string
import xgboost as xgb
import pickle
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
from collections import Counter
import matplotlib.pyplot as plt
import plotly.express as px



# preprocessing functions

# Removing Stopwords 
stop_words = set(stopwords.words('english'))

def remove_stopwords(Summary):
    # Use list comprehension for efficient list creation
    new_Summary = [word for word in Summary.split() if word not in stop_words]
    return " ".join(new_Summary)

# removing the html tags
def remove_html(review):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', review)

# removing URL and @ sign
def preprocess_text_removingq_URLand_atsign(text):
    # Remove URLs
    clean_text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r'@[^\s]+', 'user', clean_text)
    # Other preprocessing steps like removing punctuation, converting to lowercase, etc.
    return text

# expanding the contractions (is-nots)
def expand(text):
    # Expand contractions
    expanded_text = contractions.fix(text)
    return expanded_text

# punctuation
exclude = string.punctuation
# remove punctuations
def remove_punctuations(text):
    return text.translate(str.maketrans('','',exclude))

# tokenize the text
def tokenize_text(text):
    return word_tokenize(text)

# lemmatization
# Create Lemmatizer and Stemmer.
word_lem = WordNetLemmatizer()
# function
def lem_words(text):
    return [word_lem.lemmatize(word) for word in text]

#  Loading the vectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Prediction
def prediction(text):
    model = pickle.load(open('xgboost.pkl', 'rb'))
    text = [text]
    results = vectorizer.transform(text)
    predict = model.predict(results)
    if predict == 1:
        predict = 'Positive'
    elif predict == 2:
        predict = 'Neutral'
    else:
        predict = 'Negative'
    return predict



def plot_top_words_per_label(df):
    grouped = df.groupby('Sentiments')
    top_words = {}
    for label, group in grouped:
        summaries = group['Summary'].tolist()
        words = ' '.join(summaries).split()
        word_counts = Counter(words)
        top_words[label] = dict(word_counts.most_common(10))
    for label, words in top_words.items():
        plt.figure()
        plt.bar(words.keys(), words.values())
        plt.title(f'Top 10 Words for {label} Sentiments')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.show()


# Main Function
def main():
    st.title('Sentiments Analysis')
    upl = st.file_uploader('Upload file')
    if upl:
        df = pd.read_csv(upl)
        df['Summary'] = df['Summary'].str.lower()
        df['Summary'] = df['Summary'].astype(str).apply(remove_stopwords)
        df['Summary'] = df['Summary'].apply(remove_html)
        df['Summary'] = df['Summary'].apply(preprocess_text_removingq_URLand_atsign)
        df['Summary'] = df['Summary'].apply(expand)
        df['Summary'] = df['Summary'].apply(remove_punctuations)
        df['Summary'] = df['Summary'].apply(tokenize_text)
        df['Summary'] = df['Summary'].apply(lem_words)
        df['Summary'] = df['Summary'].str.join(" ")
        df['Sentiments'] = df['Summary'].apply(prediction)
        st.write(df.head(10))

        fig = px.histogram(df, x=df.Sentiments)
        fig.update_layout(
            title=f'Count Plot for Sentiments',
            xaxis_title='Sentiments',
            yaxis_title='Count')

        st.plotly_chart(fig)

        grouped = df.groupby('Sentiments')
        top_words = {}
        for label, group in grouped:
            summaries = group['Summary'].tolist()
            words = ' '.join(summaries).split()
            word_counts = Counter(words)
            top_words[label] = dict(word_counts.most_common(10))
        for label, words in top_words.items():
            fig = px.bar(x=list(words.keys()), y=list(words.values()))
            fig.update_layout(
                title=f'Top 10 Words for {label} Sentiments',
                xaxis_title='Words',
                yaxis_title='Frequency',
                xaxis_tickangle=-45
            )
        st.plotly_chart(fig)


    

        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
            )

    

if __name__ == '__main__':
    main()





