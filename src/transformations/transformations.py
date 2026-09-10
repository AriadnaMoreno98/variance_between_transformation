import numpy as np

from src.transformations.base import BaseTransformation


class DiscriminativeProjection(BaseTransformation):
    """z1 = proyeccion sobre el vector discriminativo w = (mean1-mean0)/||mean1-mean0||
    z2 = norma del residual (energia ortogonal a w)."""

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.w = None

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        X0 = X_train[y_train == 0]
        X1 = X_train[y_train == 1]

        mean_0 = X0.mean(axis=0)
        mean_1 = X1.mean(axis=0)

        w = mean_1 - mean_0
        self.w = w / (np.linalg.norm(w) + self.epsilon)
        return self

    def transform(self, X):
        X = np.asarray(X)

        z1 = X @ self.w
        projection = np.outer(z1, self.w)
        residual = X - projection
        z2 = np.linalg.norm(residual, axis=1)

        return np.column_stack((z1, z2))


class PolarityProjection(BaseTransformation):
    """z1 = media de las features con score positivo, z2 = media de las
    features con score negativo. Vector de ceros si no hay features de ese signo."""

    def __init__(self, scores=None):
        self.scores = scores

    def fit(self, X_train, y_train):
        return self

    def transform(self, X):
        X = np.asarray(X)
        self.scores = np.asarray(self.scores)

        pos_idx = np.where(self.scores > 0)[0]
        neg_idx = np.where(self.scores < 0)[0]

        z1 = X[:, pos_idx].mean(axis=1) if len(pos_idx) > 0 else np.zeros(X.shape[0])
        z2 = X[:, neg_idx].mean(axis=1) if len(neg_idx) > 0 else np.zeros(X.shape[0])

        return np.column_stack((z1, z2))


class WeightedProjection(BaseTransformation):
    """w = |scores| / sum(|scores|); z1 = combinacion lineal ponderada,
    z2 = dispersion ponderada alrededor de z1."""

    def __init__(self, scores=None, epsilon=1e-8):
        self.scores = scores
        self.epsilon = epsilon

    def fit(self, X_train, y_train):
        return self

    def transform(self, X):
        X = np.asarray(X)
        self.scores = np.asarray(self.scores)

        weights = np.abs(self.scores)
        weights = weights / (np.sum(weights) + self.epsilon)

        z1 = X @ weights
        mean_w = z1[:, None]
        z2 = np.sqrt(np.sum(weights * (X - mean_w) ** 2, axis=1))

        return np.column_stack((z1, z2))


class SplitHalvesProjection(BaseTransformation):
    """z1 = media de la primera mitad de features, z2 = media de la segunda mitad."""

    def fit(self, X_train, y_train):
        return self

    def transform(self, X):
        X = np.asarray(X)
        split_idx = X.shape[1] // 2

        z1 = X[:, :split_idx].mean(axis=1)
        z2 = X[:, split_idx:].mean(axis=1)

        return np.column_stack((z1, z2))


class VarianceBetweenProjection(BaseTransformation):
    """v = (std1-std0) normalizado, w = (mean1-mean0) normalizado.
    z1 = proyeccion sobre v, z2 = proyeccion sobre w."""

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.v = None
        self.w = None

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        X0 = X_train[y_train == 0]
        X1 = X_train[y_train == 1]

        v = X1.std(axis=0) - X0.std(axis=0)
        w = X1.mean(axis=0) - X0.mean(axis=0)

        self.v = v / (np.linalg.norm(v) + self.epsilon)
        self.w = w / (np.linalg.norm(w) + self.epsilon)
        return self

    def transform(self, X):
        X = np.asarray(X)
        z1 = X @ self.v
        z2 = X @ self.w
        return np.column_stack((z1, z2))


class VarianceBetweenBalancedProjection(BaseTransformation):
    """v y w normalizados por dispersion intra-clase (estilo Cohen's d),
    en vez de diferencia cruda. z1 = proyeccion sobre v, z2 = proyeccion sobre w."""

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.v = None
        self.w = None

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        X0 = X_train[y_train == 0]
        X1 = X_train[y_train == 1]

        std0 = X0.std(axis=0)
        std1 = X1.std(axis=0)
        denom = std0 + std1 + self.epsilon

        w = (X1.mean(axis=0) - X0.mean(axis=0)) / denom
        v = (std1 - std0) / denom

        self.v = v / (np.linalg.norm(v) + self.epsilon)
        self.w = w / (np.linalg.norm(w) + self.epsilon)
        return self

    def transform(self, X):
        X = np.asarray(X)
        z1 = X @ self.v
        z2 = X @ self.w
        return np.column_stack((z1, z2))


class CentroidDistanceProjection(BaseTransformation):
    """z1 = distancia euclidiana de cada muestra a mean1,
    z2 = distancia euclidiana de cada muestra a mean0."""

    def __init__(self):
        self.mean_0 = None
        self.mean_1 = None

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        self.mean_0 = X_train[y_train == 0].mean(axis=0)
        self.mean_1 = X_train[y_train == 1].mean(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        z1 = np.linalg.norm(X - self.mean_1, axis=1)
        z2 = np.linalg.norm(X - self.mean_0, axis=1)
        return np.column_stack((z1, z2))
