from sklearn.base import BaseEstimator, TransformerMixin
import spacy
import re
import numpy as np
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from typing import List, Union


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer for cleaning and preprocessing English text data.

    This transformer performs the following preprocessing steps:
    - Lowercasing text
    - Removing URLs and non-alphabetic characters
    - Tokenizing and lemmatizing using spaCy
    - Removing stopwords and short tokens (<= 2 characters)

    Attributes:
        batch_size (int): The number of texts processed at once in batch mode. Helps manage memory usage.
        nlp (Language): spaCy language model used for tokenization and lemmatization.

    Parameters:
        batch_size (int, optional): Size of batches for text processing. Defaults to 500.

    Example:
        >>> cleaner = TextCleaner()
        >>> texts = ["This is an example sentence!", "Visit https://example.com for more info."]
        >>> cleaned = cleaner.fit_transform(texts)
        >>> print(cleaned)
        ['example sentence', 'visit info']
    """
    def __init__(self, batch_size=500):
        self.batch_size = batch_size
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|[^a-z\s]", " ", text)
        return text

    def process_batch(self, texts: List[str]) -> List[str]:
        cleaned_texts = [self.clean_text(text) for text in texts]
        docs = list(self.nlp.pipe(cleaned_texts))
        final_texts = []

        for doc in docs:
            tokens = [
                token.lemma_
                for token in doc
                if token.lemma_ not in STOP_WORDS and len(token.lemma_) > 2
            ]
            final_texts.append(" ".join(tokens))

        return final_texts

    def fit(self, X: Union[pd.Series, np.ndarray, List[str]], y=None)  -> "TextCleaner":
        return self

    def transform(self, X: Union[pd.Series, np.ndarray, List[str]]) -> List[str]:
        if isinstance(X, pd.Series):
            X = X.tolist()
        elif isinstance(X, np.ndarray):
            X = X.tolist()

        cleaned = self._transform_in_batches(X)
        return cleaned

    def _transform_in_batches(self, X: List[str]) -> List[str]:
        output = []
        for i in range(0, len(X), self.batch_size):
            batch = X[i : i + self.batch_size]
            processed = self.process_batch(batch)
            output.extend(processed)
        return output
