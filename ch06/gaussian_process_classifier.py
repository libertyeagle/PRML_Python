import numpy as np


class GaussianProcessClassifier:

    def __init__(self, kernel, pseudo_noise_level=1e-4):
        """
        construct gaussian process classifier
        :param kernel: kernel function
        :param pseudo_noise_level: noise level (\upsilon)
        """
        self.kernel = kernel
        self.pseudo_noise_level = pseudo_noise_level

    def _sigmoid(self, a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def fit(self, X, t, iter_max=None, learning_rate=0.1):
        """
        fit the model using ILSR & gradient ascent
        :param X: (sample_size, n_features)
        :param t: (sample_size, )
        :param iter_max: max iterations, set to None will not learn hyperparameters
        :param learning_rate: initial learning rate
        :return:
        """
        if X.ndim == 1:
            X = X[:, None]
        t = t.reshape(-1)
        # (N, )
        self.X = X
        self.t = t
        Gram = self.kernel(X, X)
        I = np.eye(X.shape[0])
        self.covariance = Gram + I * self.pseudo_noise_level
        self.precision = np.linalg.inv(self.covariance)
        if (iter_max is None):
            return
        a_N = np.zeros(X.shape[0])
        sigmoid_N = self._sigmoid(a_N)
        w_N = np.diagflat(sigmoid_N)
        # (N, N)
        last_distance = np.Inf
        a_N_optimal = None
        for i in range(iter_max):
            a_N = self.covariance.dot(np.linalg.inv(I + w_N.dot(self.covariance))).dot(t - sigmoid_N + w_N.dot(a_N))
            # formula (6.83)
            sigmoid_N = self._sigmoid(a_N)
            dist = abs(np.sum(np.abs(a_N - self.covariance.dot(t - sigmoid_N))))
            if dist < last_distance:
                last_distance = dist
                a_N_optimal = a_N
            else:
                break
        sigmoid_N = self._sigmoid(a_N_optimal)
        w_N = np.diagflat(sigmoid_N)

        self.a_N = a_N_optimal
        self.w_N = w_N
        self.sigmoid_N = sigmoid_N

        last_loglikehood = -np.Inf

        for i in range(iter_max):
            covarience_gradients = self.kernel.derivatives(X, X)
            # (n_kernel_params, N, N)
            # notice that noise_like term does not have gradient, pseudo_noise_level is fixed
            updates_part_1 = np.array(
                [0.5 * a_N_optimal.T.dot(self.precision).dot(grad).dot(self.precision)
                    .dot(a_N_optimal)
                 - 0.5 *
                 np.trace(np.linalg.inv(I + self.covariance.dot(w_N)).dot(w_N).dot(grad))
                 for grad in covarience_gradients]
            )

            # formula (6.91)
            diag_matrix = np.diag(
                np.linalg.inv(I + self.covariance.dot(w_N)).dot(self.covariance)
            )
            # (N, )
            a_N_optimal_grad = np.array(
                [np.linalg.inv(I + self.covariance.dot(w_N)).dot(grad)
                     .dot(t - sigmoid_N)
                 for grad in covarience_gradients]
            )
            # formula (6.92)
            # (n_kernel_params, N)
            vector_to_mul = (sigmoid_N[:, None] * (1 - sigmoid_N[:, None]) *
                             (1 - 2 * sigmoid_N[:, None]) * a_N_optimal_grad.T).T
            # (n_kernel_params, N)
            updates_part_2 = - 0.5 * np.sum(diag_matrix[:, None].T * vector_to_mul, axis=-1)
            # (n_kernel_params, )
            # formula (6.93)

            updates = updates_part_1 + updates_part_2
            for j in range(iter_max):
                self.kernel.update_parameters(learning_rate * updates)
                Gram = self.kernel(X, X)
                self.covariance = Gram + I * self.pseudo_noise_level
                self.precision = np.linalg.inv(self.covariance)
                log_likelihood = self._log_likelihood()
                if log_likelihood > last_loglikehood:
                    last_loglikehood = log_likelihood
                    print("iteration {:d}: log likelihood {:f}".format(i, log_likelihood))
                    break
                else:
                    # undo
                    self.kernel.update_parameters(-learning_rate * updates)
                    learning_rate *= 0.9

    def _log_likelihood(self):
        return -0.5 * self.a_N.T.dot(self.covariance).dot(self.a_N) \
               - 0.5 * np.linalg.slogdet(self.covariance)[1] \
               + np.dot(self.t, self.a_N) \
               - np.sum(np.log(1 + np.exp(self.a_N))) \
               - 0.5 * np.linalg.slogdet(self.w_N + self.precision)[1]
        # `np.linalg.slogdet()` compute
        # the natural log of the absolute value of the determinant.

    def predict(self, X, with_error=False):
        """
        make predictions using (6.87)
        :param X:
        :param with_error:
        :return:
        """
        K = self.kernel(self.X, X)
        # shape: (training_set_size, X_sample_size)
        prediction_mean = K.T.dot(self.precision).dot(self.t - self.sigmoid_N)
        if with_error:
            variance = (
                    self.kernel(X, X, pairwise=False)
                    + self.pseudo_noise_level
                    - np.diag(K.T.dot(np.linalg.inv(np.linalg.inv(self.w_N) + self.covariance)).dot(K))
            )
            # formula (6.88)
            return self._sigmoid(prediction_mean), variance

        return self._sigmoid(prediction_mean)
