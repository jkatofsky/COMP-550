import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import argparse

# read comments, vectorization 
df = pd.read_csv("../../data/dataset.csv", lineterminator='\n')
df = df.head(100) # TODO: delete me, I'm just for testing
df = df[df["label"] != "UNKNOWN"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectorizer', type=str, required=True)
    args = parser.parse_args()

    # clustering
    vectorizer = args.vectorizer
    if vectorizer == "USE":
        from universal_sentence_encoder import universal_sentence_encoder
        X = universal_sentence_encoder(df)
    elif vectorizer == "sBERT":
        from sentence_bert import sentence_bert
        X = sentence_bert(df)
    elif vectorizer == "ngram":
        from ngram import ngram
        X = ngram(df, analyzer='word') # TODO: better way of parameterizing ngram

    kmeans = KMeans(n_clusters=11, random_state=0).fit(X) # NOTE: 11 clusters for 11 majors

    # TODO: calculate similarity!
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)