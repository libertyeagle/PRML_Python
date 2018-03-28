from base_classifier import Classifier
import numpy as np

class Perceptron(Classifier):


    def _fit(self, X, t, epoch=100):
        self._check_binary_negative(t)
        self._w = np.random.rand(X.shape[1])    # init parameter w
        num_instances = X.shape[0]
        for i in range(epoch):
            permuted = np.random.permutation(num_instances)
            batch_X = X[permuted]
            batch_t = t[permuted]
            for x, label in zip(batch_X, batch_t):
                prediction = np.sign(self._w @ x)
                if prediction != label:
                    self._w += x * label
            if (np.dot(X, self._w.T) * t > 0).all():
                break


    def _classify(self, X):
        return np.sign(X @ self._w.T).astype(np.int)
