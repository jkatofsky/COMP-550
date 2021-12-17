import pickle
import pandas as pd
from matplotlib import pyplot as plt

from sklearn import metrics

# get true labels
df = pd.read_csv("../../data/dataset.csv", lineterminator='\n')
df = df[df["label"] != "UNKNOWN"]
df.fillna("", inplace=True)
labels_true = df["label"]
labels_true = labels_true.to_numpy()

# load models
with open("../../model/kmeans-USE-12-17-17-43.pickle","rb") as f:
  USE = pickle.load(f)
with open("../../model/kmeans-sBERT-12-17-17-43.pickle","rb") as f:
  sBERT = pickle.load(f)
with open("../../model/kmeans-ngram-3-char-wb-12-17-16-58.pickle","rb") as f:
  ngram = pickle.load(f)

# score model predictions
USE_score = metrics.adjusted_mutual_info_score(labels_true, USE.labels_)
sBERT_score = metrics.adjusted_mutual_info_score(labels_true, sBERT.labels_)
ngram_score = metrics.adjusted_mutual_info_score(labels_true, ngram.labels_)


# plot result
scores = [USE_score,sBERT_score,ngram_score]
labels = ["USE", "sBERT", "ngram"]

plt.bar(range(len(scores)), scores, tick_label=labels)
plt.xlabel("Vectorization Method")
plt.ylabel("MI Score")
plt.title("MI Score across Vectoriation Methods")
plt.savefig("../../fig/MI Score across Vectoriation Methods.png")
