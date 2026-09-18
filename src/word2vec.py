from gensim.models import KeyedVectors
from sklearn.base import BaseEstimator, TransformerMixin
import gensim.downloader as api
import numpy as np
from typing import List, Union


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Transforms input text into dense vector representations using pre-trained GloVe word embeddings.

    This transformer splits each input string into tokens, retrieves the corresponding GloVe vectors 
    for known words, and returns the average vector for each text sample.

    Attributes:
        model (gensim.models.KeyedVectors): Pre-trained GloVe model loaded from gensim API.
        dim (int): Dimensionality of the GloVe vectors (e.g., 50).

    Methods:
        fit(X, y=None): Does nothing, present for compatibility with scikit-learn pipeline.
        transform(X): Transforms a list of texts into an array of dense vector representations.

    Parameters:
        None (no configurable parameters at initialization)

    Returns:
        np.ndarray: A 2D array of shape (n_samples, embedding_dim), where each row is the 
                    average word vector of the corresponding input text.

    Example:
        >>> vec = Word2VecVectorizer()
        >>> X = ["this is a test", "another sentence"]
        >>> X_vec = vec.fit_transform(X)
        >>> X_vec.shape
        (2, 50)
    """
    def __init__(self):
        self.model = api.load("glove-wiki-gigaword-50")
        self.dim = self.model.vector_size

    def fit(self, X: Union[List[str], np.ndarray], y=None) -> "Word2VecVectorizer":
        return self

    def transform(self, X: Union[List[str], np.ndarray]) -> np.ndarray:
        vectors = []
        for text in X:
            tokens = text.split()
            word_vecs = [self.model[word] for word in tokens if word in self.model]
            if word_vecs:
                vec = np.mean(word_vecs, axis=0)
            else:
                vec = np.zeros(self.dim)
            vectors.append(vec)
        return np.array(vectors)
