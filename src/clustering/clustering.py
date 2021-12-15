import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# read comments, vectorization 
df = pd.read_csv("../../data/dataset.csv", lineterminator='\n')
df = df.head(100) # TODO: delete me, I'm just for testing

# TODO: this is contextualized embedding, need n-gram vectorization too
# TODO: better code style for different vectorization methods

sBERT = True

if not sBERT:
    from universal_sentence_encoder import universal_sentence_encoder
    X = universal_sentence_encoder(df)
else:
    from sentence_bert import sentence_bert
    X = sentence_bert(df)

# clustering
kmeans = KMeans(n_clusters=11, random_state=0).fit(X) # NOTE: 11 clusters for 11 majors
print(kmeans.labels_)
print(kmeans.cluster_centers_)
