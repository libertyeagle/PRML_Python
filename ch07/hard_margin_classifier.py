import numpy as np


class HardMarginClassifier:

    def __init__(self, kernel):
        """
        constructor hard margin SVM
        :param kernel: kernel to use
        """
        self.kernel = kernel

    def fit(self, X, t, learning_rate=0.1, decay_step=10000, decay_rate=0.9, min_learning_rate=1e-5):

        if X.ndim = 1:
            X = X[:, None]

        lr = learning_rate
        t_l2_norm_square = np.sum(np.square(t))
        a = np.ones_like(t)
        Gram = self.kernel(X, X)
        gradient_coefficient = t * t[:, None] * Gram
        while True:
            for _ in range(decay_step):
                # compute gradient according to the gradient of formula (7.10)
                gradient = 1 - gradient_coefficient @ a
                # we wish to maximize (7.10), so it is plus
                a += lr * gradient
                # in order to satisfy constraint (7.12)
                a -= (a @ t) * t / t_l2_norm_square
                # ((a_orig @ t) / t_l2_norm_square * t) @ t = a_orig @ t
                a = a.clip(0)
                # less than 0 element clip to 0 in order to satisfy formula (7.11)
