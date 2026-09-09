from sklearn.feature_extraction.text import TfidfVectorizer

from src.representations.base import BaseRepresentation


class TFIDFRepresentation(BaseRepresentation):
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts).toarray()
