from abc import ABC, abstractmethod

import numpy as np


class BaseTransformation(ABC):
    """Interfaz comun para las transformaciones MML al plano 2D (z1, z2)."""

    @abstractmethod
    def fit(self, X_train, y_train):
        """Ajusta la transformacion sobre train. Devuelve self."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, X) -> np.ndarray:
        """Proyecta X (train o test) al plano 2D. Shape (n_samples, 2)."""
        raise NotImplementedError

    def fit_transform(self, X_train, y_train) -> np.ndarray:
        """Ajusta sobre train y proyecta ese mismo train."""
        self.fit(X_train, y_train)
        return self.transform(X_train)
