import pickle, os.path
import numpy as np

# https://spacy.io/universe/project/spacy-universal-sentence-encoder

def universal_sentence_encoder(df):
    filename = "../../vectors/universal-sentence-encoder.pickle"
    if os.path.isfile(filename):
        with open(filename,"rb") as f:
            X = pickle.load(f)
            return X
    else:
        import spacy_universal_sentence_encoder
        nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')
        X = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            X.append(nlp(str(row["comment"])).vector)
        X = np.array(X)
        with open(filename,"wb") as f:
            pickle.dump(X,f)
        return X
        
