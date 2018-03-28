import numpy as np

class Classifier():
    """
    base classifier
    """
    def fit(self, X, t, **kwargs):
        """
        :param X: training set input X of (sample_size, ndim) ndarray
        :param t: training set target t of (sample_size, ) ndarray
        :param kwargs:
        :return:
        """
        self._check_input(X)
        self._check_target(t)

        if hasattr(self, "_fit"):
            self._fit(X, t, **kwargs)
        else:
            raise NotImplementedError

    def classify(self, X, **kwargs):
        """
        :param X: sample set of X to classify, shape of (sample_size, ndim)
        :param kwargs:
        :return:
        label (sample_size,)
        """
        self._check_input(X)

        if hasattr(self, "_classify"):
            return self._classify(X, **kwargs)
        else:
            raise NotImplementedError

    def proba(self, X, **kwargs):
        """
        :param X: sample set of X to classify, shape of (sample_size, ndim)
        :param kwargs:
        :return:
        probability (sample_size, n_classes)
        """
        self._check_input(X)
        if hasattr(self, "_proba"):
            return self._proba(X, **kwargs)
        else:
            raise NotImplementedError

    def _check_input(self, X):
        # check if training set X valid
        if not isinstance(X, np.ndarray):
            raise ValueError("X(input) must be np.ndarray")
        if X.ndim != 2:
            raise ValueError("X(input) must be two dimensional array")
        # given input set X doesn't have the same number of features as the classifier
        if hasattr(self, "n_features") and self.n_features != np.size(X, 1):
            raise ValueError(
                "mismatch in dimension 1 of X(input) (size {} is different from {})"
                .format(np.size(X, 1), self.n_features)
            )

    def _check_target(self, t):
        if not isinstance(t, np.ndarray):
            raise ValueError("t(target) must be np.ndarray")
        if t.ndim != 1:
            raise ValueError("t(target) must be one dimensional array")
        if t.dtype != np.int:
            raise ValueError("dtype of t(target) must be np.int")


    # for binary classification where class label is 0 / 1
    def _check_binary(self, t):
        n_zeros = np.count_nonzero(t == 0)
        n_ones = np.count_nonzero(t == 1)
        if n_zeros + n_ones != t.size:
            raise ValueError("t(target) must only has 0 or 1")

    # for binary classification where class label is -1 / 1 (e.g, Perceptron)
    def _check_binary_negative(self, t):
        n_zeros = np.count_nonzero(t == -1)
        n_ones = np.count_nonzero(t == 1)
        if n_zeros + n_ones != t.size:
            raise ValueError("t(target) must only has -1 or 1")