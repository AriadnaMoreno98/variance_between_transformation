import numpy as np


class MMLPipeline:
    """Pipeline MML basado en la arquitectura OOP (representations/selection/
    transformations/search).

    `representation` es opcional: si se omite (None), fit/predict/get_projection
    asumen que X_train/X_test ya vienen vectorizados (util cuando la
    representacion se calcula una sola vez por fuera y se comparte entre
    varios pipelines)."""

    def __init__(self, *, selector, transformation, search, representation=None, k=100):
        self.representation = representation
        self.selector = selector
        self.transformation = transformation
        self.search = search
        self.k = k
        self.selected_idx = None

    def fit(self, X_train, y_train):
        X_rep = self.representation.fit_transform(X_train) if self.representation is not None else X_train

        self.selector.fit(X_rep, y_train)
        self.selected_idx = self.selector.top_k(self.k)

        X_sel = X_rep[:, self.selected_idx]

        # Algunas transformaciones (polarity/weighted/...) necesitan los scores
        # del selector, alineados con las features seleccionadas. Se inyectan
        # aqui porque solo se conocen despues de selector.fit(), y la
        # transformacion ya viene instanciada.
        if hasattr(self.transformation, "scores"):
            self.transformation.scores = self.selector.scores()[self.selected_idx]

        Z_train = self.transformation.fit_transform(X_sel, y_train)
        self.search.fit(Z_train, y_train)
        return self

    def predict(self, X_test) -> np.ndarray:
        Z_test = self.get_projection(X_test)
        return self.search.predict(Z_test)

    def fit_predict(self, X_train, y_train, X_test) -> np.ndarray:
        self.fit(X_train, y_train)
        return self.predict(X_test)

    def boundary_params(self) -> dict:
        return self.search.boundary_params()

    def get_projection(self, X) -> np.ndarray:
        """Proyecta X al plano 2D (z1, z2) sin clasificar, util para visualizacion."""
        X_rep = self.representation.transform(X) if self.representation is not None else X
        X_sel = X_rep[:, self.selected_idx]
        return self.transformation.transform(X_sel)
