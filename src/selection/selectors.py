import numpy as np

from src.selection.base import BaseSelector


def compute_dmeans_scores(X_train, y_train):
    y_train = np.asarray(y_train)

    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]

    mean_0 = np.asarray(X0.mean(axis=0)).ravel()
    mean_1 = np.asarray(X1.mean(axis=0)).ravel()

    scores = mean_1 - mean_0
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


class DMeansSelector(BaseSelector):
    def __init__(self):
        self._scores = None

    def fit(self, X, y):
        self._scores = compute_dmeans_scores(X, y)
        return self

    def scores(self):
        return self._scores
