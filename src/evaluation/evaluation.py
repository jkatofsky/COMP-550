import pickle
import pandas as pd
from matplotlib import pyplot as plt

from sklearn import metrics

# get true labels
df = pd.read_csv("../../data/dataset.csv", lineterminator='\n')
to_filter = ['UNKNOWN', 'Management', 'Education', 'Law', 'Music']
df = df[~df['label'].isin(to_filter)].reset_index()
df.fillna("", inplace=True)
labels_true = df["label"]
labels_true = labels_true.to_numpy()

# load models
with open("../../model/kmeans-USE-12-19-19-56.pickle", "rb") as f:
  USE = pickle.load(f)
with open("../../model/kmeans-sBERT-12-19-19-57.pickle", "rb") as f:
  sBERT = pickle.load(f)
with open("../../model/kmeans-ngram-3-char-wb-12-19-20-02.pickle", "rb") as f:
  ngram = pickle.load(f)

# score model predictions
# USE_score = metrics.adjusted_mutual_info_score(labels_true, USE.labels_)
# sBERT_score = metrics.adjusted_mutual_info_score(labels_true, sBERT.labels_)
# ngram_score = metrics.adjusted_mutual_info_score(_labels_true, ngram.labels_)

USE_score = metrics.rand_score(labels_true, USE.labels_)
sBERT_score = metrics.rand_score(labels_true, sBERT.labels_)
ngram_score = metrics.rand_score(labels_true, ngram.labels_)

# plot result
scores = [USE_score,sBERT_score,ngram_score]
labels = ["USE", "sBERT", "ngram"]

plt.bar(range(len(scores)), scores, tick_label=labels)
plt.xlabel("Vectorization Method")
plt.ylabel("Rand Index")
plt.title("Rand Index across Vectorization Methods")
plt.savefig("../../fig/Rand Score across Vectorization Methods.png")
