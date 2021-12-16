import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def get_sw():
    nltk.download('stopwords',quiet=True)
    sw = stopwords.words('english')
    # TODO: add task-specific stopwords (ex. BIOL200, COMP512)
    # NOTE: sw is a list
    return sw

def ngram(df, ngram_range=(5,5), analyzer='word'):
    # NOTE: analyzer{‘word’, ‘char’, ‘char_wb’} or callable, default=’word’
    #       ‘char’ is character-by-character ngram
    vectorizer = CountVectorizer(stop_words=get_sw(), ngram_range=ngram_range, analyzer=analyzer)
    X = vectorizer.fit_transform(df["comment"])
    print(vectorizer.get_feature_names_out()) 
    return X
