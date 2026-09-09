import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE


def get_baseline_models():
    return {
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "svm": LinearSVC(random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "nb": MultinomialNB(),
        "lr_balanced": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "svm_balanced": LinearSVC(random_state=42, class_weight="balanced"),
        "mlp": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    }


PARAM_GRIDS = {
    "logreg": {"C": [0.01, 0.1, 1, 10]},
    "lr_balanced": {"C": [0.01, 0.1, 1, 10]},
    "svm": {"C": [0.01, 0.1, 1, 10]},
    "svm_balanced": {"C": [0.01, 0.1, 1, 10]},
    "knn": {"n_neighbors": [3, 5, 7, 10]},
    "mlp": {"alpha": [0.0001, 0.001, 0.01]},
}


class BaselinePipeline:
    """Pipeline baseline generico: cualquier BaseRepresentation + un modelo
    clasico (logreg/svm/knn/nb/lr_balanced/svm_balanced/mlp).

    `representation` es opcional: si se omite (None), fit/predict asumen que
    X_train/X_test ya vienen vectorizados (util cuando la representacion se
    calcula una sola vez por fuera y se comparte entre varios pipelines).

    `use_smote=True` aplica SMOTE sobre (X_train, y_train) antes de entrenar
    (solo en fit/fit_with_grid_search, nunca en predict)."""

    def __init__(self, *, model_name, representation=None, use_smote=False):
        models = get_baseline_models()

        if model_name not in models:
            raise ValueError(f"Modelo no soportado: {model_name}")

        self.representation = representation
        self.use_smote = use_smote
        self._model_name = model_name
        self.model = models[model_name]
        self.best_params_ = None

    def _vectorize_train(self, X_train):
        X_rep = self.representation.fit_transform(X_train) if self.representation is not None else X_train

        if self._model_name == "nb":
            X_rep = np.clip(X_rep, 0, None)

        return X_rep

    def fit(self, X_train, y_train):
        X_rep = self._vectorize_train(X_train)

        if self.use_smote:
            X_rep, y_train = SMOTE(random_state=42).fit_resample(X_rep, y_train)

        self.model.fit(X_rep, y_train)
        return self

    def fit_with_grid_search(self, X_train, y_train, cv=3):
        """Ajusta el modelo con GridSearchCV (scoring=balanced_accuracy) sobre
        el grid de hiperparametros correspondiente al modelo. Para nb, que no
        tiene grid definido, hace un fit normal. Reporta e imprime los
        mejores hiperparametros encontrados (self.best_params_)."""
        X_rep = self._vectorize_train(X_train)

        if self.use_smote:
            X_rep, y_train = SMOTE(random_state=42).fit_resample(X_rep, y_train)

        param_grid = PARAM_GRIDS.get(self._model_name)

        if param_grid is None:
            self.model.fit(X_rep, y_train)
            print(f"[{self._model_name}] no tiene grid de hiperparametros definido, fit normal.")
            return self

        grid_search = GridSearchCV(
            self.model,
            param_grid=param_grid,
            cv=cv,
            scoring="balanced_accuracy",
        )
        grid_search.fit(X_rep, y_train)

        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_

        print(
            f"[{self._model_name}] mejores hiperparámetros: {grid_search.best_params_} "
            f"| best_balanced_accuracy_cv={grid_search.best_score_:.4f}"
        )
        return self

    def predict(self, X_test) -> np.ndarray:
        X_rep = self.representation.transform(X_test) if self.representation is not None else X_test

        if self._model_name == "nb":
            X_rep = np.clip(X_rep, 0, None)

        return self.model.predict(X_rep)

    def fit_predict(self, X_train, y_train, X_test) -> np.ndarray:
        self.fit(X_train, y_train)
        return self.predict(X_test)

    def model_name(self) -> str:
        return self._model_name
