import pickle
import argparse
from datetime import datetime

import pandas as pd
from sklearn.cluster import KMeans


# read comments, vectorization 
df = pd.read_csv("../../data/dataset.csv", lineterminator='\n')
df = df[df["label"] != "UNKNOWN"]
df.fillna("", inplace=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectorizer', type=str, required=True)
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--ngram_type', type=str, default="char_wb")
    args = parser.parse_args()

    # vectorization
    vectorizer = args.vectorizer
    if vectorizer == "USE":
        from universal_sentence_encoder import universal_sentence_encoder
        X = universal_sentence_encoder(df)
    elif vectorizer == "sBERT":
        from sentence_bert import sentence_bert
        X = sentence_bert(df)
    elif vectorizer == "ngram":
        from ngram import ngram
        X = ngram(df, analyzer=args.ngram_type, ngram_range=(args.ngram_size,args.ngram_size)) 
        vectorizer = "ngram-" + str(args.ngram_size) + "-" + str(args.ngram_type)

    # clustering
    km = KMeans(n_clusters=11, random_state=0).fit(X)

    # save model
    now = datetime.now()
    filename = "../../model/kmeans-" + vectorizer.replace("_","-") + "-" + now.strftime("%m-%d-%H-%M") + ".pickle"
    with open(filename, "wb") as f:
        pickle.dump(km,f)
