import numpy as np

from src.search.base import BaseSearch


def find_best_threshold(z2, y_true):
    """
    Fuerza bruta exacta para línea horizontal z2 = t,
    optimizada con acumulados.
    """
    z2 = np.asarray(z2)
    y_true = np.asarray(y_true)

    order = np.argsort(z2)
    z_sorted = z2[order]
    y_sorted = y_true[order]

    n = len(y_sorted)
    total_pos = np.sum(y_sorted == 1)
    total_neg = np.sum(y_sorted == 0)

    if total_pos == 0 or total_neg == 0:
        return float(z_sorted[0]), "greater", 0.5

    candidates = []

    # corte antes del primero
    candidates.append((z_sorted[0] - 1e-12, 0))

    # cortes entre valores distintos
    for i in range(n - 1):
        if z_sorted[i] != z_sorted[i + 1]:
            t = (z_sorted[i] + z_sorted[i + 1]) / 2.0
            candidates.append((t, i + 1))

    # corte después del último
    candidates.append((z_sorted[-1] + 1e-12, n))

    best_score = -1.0
    best_t = None
    best_orientation = None

    for t, left_count in candidates:
        left_y = y_sorted[:left_count]
        right_y = y_sorted[left_count:]

        # orientación greater:
        # derecha/arriba = clase 1, izquierda/abajo = clase 0
        tp = np.sum(right_y == 1)
        tn = np.sum(left_y == 0)

        recall_pos = tp / total_pos
        recall_neg = tn / total_neg
        ba_greater = (recall_pos + recall_neg) / 2.0

        if ba_greater > best_score:
            best_score = ba_greater
            best_t = t
            best_orientation = "greater"

        # orientación less_equal:
        # izquierda/abajo = clase 1, derecha/arriba = clase 0
        tp = np.sum(left_y == 1)
        tn = np.sum(right_y == 0)

        recall_pos = tp / total_pos
        recall_neg = tn / total_neg
        ba_less_equal = (recall_pos + recall_neg) / 2.0

        if ba_less_equal > best_score:
            best_score = ba_less_equal
            best_t = t
            best_orientation = "less_equal"

    return float(best_t), best_orientation, float(best_score)


class HorizontalSearch(BaseSearch):
    """Metodo actual del pipeline: linea horizontal z2 = t, buscada por
    fuerza bruta exacta (find_best_threshold)."""

    def __init__(self):
        self.threshold = None
        self.orientation = None
        self.train_score = None

    def fit(self, Z_train, y_train):
        z2 = np.asarray(Z_train)[:, 1]
        self.threshold, self.orientation, self.train_score = find_best_threshold(z2, y_train)
        return self

    def predict(self, Z_test):
        z2 = np.asarray(Z_test)[:, 1]

        if self.orientation == "greater":
            return (z2 > self.threshold).astype(int)
        if self.orientation == "less_equal":
            return (z2 <= self.threshold).astype(int)
        raise ValueError(f"Orientación no válida: {self.orientation}")

    def boundary_params(self):
        return {
            "theta": 90.0,
            "threshold": self.threshold,
            "orientation": self.orientation,
        }
