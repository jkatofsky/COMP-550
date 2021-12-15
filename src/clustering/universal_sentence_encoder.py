import numpy as np
import spacy_universal_sentence_encoder

# https://spacy.io/universe/project/spacy-universal-sentence-encoder

def universal_sentence_encoder(dataset):
    nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')
    X = []
    for index, row in dataset.iterrows():
        if row["label"] != "UNKNOWN":
            X.append(nlp(row["comment"]).vector)
    return np.array(X)
