import numpy as np
from base_classifier import Classifier

class logisiticRegression(Classifier):
    """
    4.3.2 & 4.3.3
    logistic regression using iterative reweighted least squares
    """

    def _fit(self, X, t, max_epoch=1000):
        self._check_binary(t)
        self._w = np.random.rand(X.shape[1])
        for _ in range(max_epoch):
            w_prev = np.copy(self._w)
            y = self._sigmoid(X @ self._w.T)
            grad = np.dot(X.T, y - t)                   # according to (4.96)
            hessian = X.T @ np.diag(y * (1 - y)) @ X    # according to (4.97)
            self._w -= np.linalg.pinv(hessian) @ grad
            if np.allclose(self._w, w_prev):
                break


    def _sigmoid(self, x):
        return np.divide(1, 1 + np.exp(-x))

    def _classify(self, X, threshold=0.5):
        return (self._sigmoid(X @ self._w.T) > threshold).astype(np.int)

    def _proba(self, X):
        return self._sigmoid(X @ self._w.T)