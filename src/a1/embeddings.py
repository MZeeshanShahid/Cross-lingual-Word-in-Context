import gensim

def load_embedding(modelfile):
    if modelfile.endswith("parameters.bin"):
        return gensim.models.fasttext.load_facebook_vectors(modelfile)
    elif modelfile.endswith(".bin.gz") or modelfile.endswith(".bin"):
        return gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=True, unicode_errors="replace"
        )
    elif any(modelfile.endswith(ext) for ext in [".txt.gz", ".txt", ".vec.gz", ".vec"]):
        return gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=False, unicode_errors="replace"
        )
    else:
        return gensim.models.KeyedVectors.load(modelfile)