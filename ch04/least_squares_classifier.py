from base_classifier import Classifier
import numpy as np

class leastSquaresClassifier(Classifier):

    # this is a multi-class Classifier

    def _fit(self, X, t):
        # here we create an identity matrix to represent all possible one-hot enconding for K (actually, K+1 classes)
        # we need one extra dimension because label is from 1..K, not from 0 which python uses
        # and vector t is used to select corresponding one-hot enconding target vector for each instance
        T = np.eye(int(np.max(t)) + 1)[t]
        # calculate parameter closed-form solution
        self._W = np.dot(np.linalg.pinv(X), T)

    def _classify(self, X):
        # XW is a matrix of shape (N, K)
        return np.argmax(np.dot(X, self._W), axis=1).astype(np.int)
