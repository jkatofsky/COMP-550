import numpy as np
import pandas as pd
import spacy_universal_sentence_encoder, spacy_sentence_bert
from sklearn.cluster import KMeans

sBERT = False

if not sBERT:
    # https://spacy.io/universe/project/spacy-universal-sentence-encoder
    nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')
else:
    # TODO: debug sentence-BERT usage
    # https://spacy.io/universe/project/spacy-sentence-bert
    nlp = spacy_sentence_bert.load_model('en_stsb_roberta_base')

# read comments, vectorization 
# TODO: this is contextualized embedding, need n-gram vectorization too
df = pd.read_csv("../../data/dataset.csv", lineterminator='\n')
X = []
for index, row in df.iterrows():
    # TODO: only append if label is not UNKNOWN (otherwise cannot evaluate)
    X.append(nlp(row["comment"]).vector)
    if index > 100: # TODO: remove me! (just for test run)
        break

# clustering
X = np.array(X)
kmeans = KMeans(n_clusters=11, random_state=0).fit(X) # NOTE: 11 clusters for 11 majors
print(kmeans.labels_)
print(kmeans.cluster_centers_)
