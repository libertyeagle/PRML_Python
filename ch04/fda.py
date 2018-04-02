import numpy as np
from base_classifier import Classifier

class fisherDiscriminantAnalysis(Classifier):

    def _fit(self, X, t, clip_min_norm=1e-10):
        self._check_binary(t)
        X0 = X[t == 0]
        X1 = X[t == 1]
        m0 = np.mean(X0, axis=0)        # m1
        m1 = np.mean(X1, axis=0)        # m2
        within_class_cov = np.dot((X0 - m0).T, X0 - m0) + np.dot((X1 - m1).T, X1 - m1)   # equation (4.28)
        self._w = np.dot(np.linalg.pinv(within_class_cov), m1 - m0)     # equation (4.30)
        self._w /= np.linalg.norm(self._w).clip(min=clip_min_norm)
        # ensure self._w
        self._threshold = np.dot(self._w, 1/2 * (m0 + m1))
        # threshold: set to hyperplane between projections of the two means,

    def _classify(self, X):
        return (np.dot(X, self._w.T) > self._threshold).astype(np.int)