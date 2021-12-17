import pickle, os.path
from tqdm import tqdm

import numpy as np
import spacy_sentence_bert

# https://spacy.io/universe/project/spacy-sentence-bert

def sentence_bert(df):
    filename = "../../vectors/sentence-bert.pickle"
    if os.path.isfile(filename):
        with open(filename,"rb") as f:
            X = pickle.load(f)
            return X
    else:
        nlp = spacy_sentence_bert.load_model('en_stsb_roberta_base')
        X = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            X.append(nlp(str(row["comment"])).vector)
        X = np.array(X)
        with open(filename,"wb") as f:
            pickle.dump(X,f)
        return X
