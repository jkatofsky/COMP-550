import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics

# read dataset
df = pd.read_csv("../../data/dataset.csv", lineterminator='\n')
df = df[df["label"] != "UNKNOWN"]
df.fillna("", inplace=True)
df = df.head(1000)

# get true labels
labels_true = df["label"]
labels_true = labels_true.to_numpy()

# params 
ngram_ranges = [(i,i) for i in range(1,11)]
analyzers = ['word', 'char', 'char_wb']


# score each param combination
tags = []
preds = []
scores = []

nltk.download('stopwords',quiet=True)
sw = stopwords.words('english')

for ngram_range in ngram_ranges:
  for analyzer in analyzers:
    tags.append(str(ngram_range[0]) + "-" + analyzer)
    
    vectorizer = CountVectorizer(stop_words=sw, ngram_range=ngram_range, analyzer=analyzer)
    X = vectorizer.fit_transform(df["comment"])
    
    km = KMeans(n_clusters=11, random_state=0).fit(X)
    
    labels_pred = km.labels_
    preds.append(labels_pred)
    
    score = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    scores.append(score)


# word figure
word_scores = scores[::3]
plt.plot(np.arange(1,11),word_scores,label="word")

# char figure
char_scores = scores[1::3]
plt.plot(np.arange(1,11),char_scores,label="char")

# char_wb figure
char_wb_scores = scores[2::3]
plt.plot(np.arange(1,11),char_wb_scores,label="char_wb")

# save figure
plt.legend()
plt.xticks(np.arange(1,11))
plt.xlabel("Ngram Range")
plt.ylabel("Adjusted Mutual Information")
plt.title("Ngram Param MI Scores")
plt.savefig("Ngram Param MI Scores.png")
