import numpy as np
import spacy_sentence_bert

# https://spacy.io/universe/project/spacy-sentence-bert

def sentence_bert(dataset):
    nlp = spacy_sentence_bert.load_model('en_stsb_roberta_base')
    X = []
    for index, row in dataset.iterrows():
        if row["label"] != "UNKNOWN":
            X.append(nlp(row["comment"]).vector)
    return np.array(X)
