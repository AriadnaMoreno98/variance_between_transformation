from abc import ABC, abstractmethod

import numpy as np


class BaseSearch(ABC):
    """Interfaz comun para los metodos de busqueda del limite de decision
    sobre el plano 2D (z1, z2)."""

    @abstractmethod
    def fit(self, Z_train, y_train):
        """Ajusta el limite de decision sobre train. Devuelve self."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, Z_test) -> np.ndarray:
        """Clasifica Z_test (0/1) segun el limite encontrado."""
        raise NotImplementedError

    @abstractmethod
    def boundary_params(self) -> dict:
        """Parametros del limite de decision encontrado."""
        raise NotImplementedError
