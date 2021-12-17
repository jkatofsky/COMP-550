import pickle, os.path

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

def get_sw():
    nltk.download('stopwords',quiet=True)
    sw = stopwords.words('english')
    # TODO: add task-specific stopwords (ex. BIOL200, COMP512)
    # NOTE: sw is a list
    return sw

def ngram(df, ngram_range=(3,3), analyzer='char_wb'):
    # NOTE: analyzer{‘word’, ‘char’, ‘char_wb’} or callable, default=’word’
    #       ‘char’ is character-by-character ngram
    params = ["ngram",str(ngram_range[0]),analyzer]
    filename = "../../vectors/" + "-".join(params) + ".pickle"
    if os.path.isfile(filename):
        with open(filename,"rb") as f:
            X = pickle.load(f)
            return X
    else:
        vectorizer = CountVectorizer(stop_words=get_sw(), ngram_range=ngram_range, analyzer=analyzer)
        X = vectorizer.fit_transform(df["comment"])
        with open(filename,"wb") as f:
            pickle.dump(X,f)
        return X
