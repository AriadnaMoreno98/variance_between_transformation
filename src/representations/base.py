from abc import ABC, abstractmethod

import numpy as np


class BaseRepresentation(ABC):
    """Interfaz comun para todas las representaciones de texto."""

    @abstractmethod
    def fit(self, texts):
        """Ajusta la representacion sobre los textos dados. Devuelve self."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, texts) -> np.ndarray:
        """Transforma los textos a su representacion vectorial."""
        raise NotImplementedError

    def fit_transform(self, texts) -> np.ndarray:
        """Ajusta y transforma en un solo paso."""
        self.fit(texts)
        return self.transform(texts)
