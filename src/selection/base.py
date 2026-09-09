from abc import ABC, abstractmethod

import numpy as np


class BaseSelector(ABC):
    """Interfaz comun para todos los metodos de seleccion de features."""

    @abstractmethod
    def fit(self, X, y):
        """Ajusta el selector sobre X, y. Devuelve self."""
        raise NotImplementedError

    @abstractmethod
    def scores(self) -> np.ndarray:
        """Scores crudos por feature (mayor = mas relevante)."""
        raise NotImplementedError

    def rank(self) -> np.ndarray:
        """Indices de features ordenados de mayor a menor relevancia."""
        return np.argsort(np.abs(self.scores()))[::-1]

    def top_k(self, k) -> np.ndarray:
        """Primeros k indices segun rank()."""
        return self.rank()[:k]
