import numpy as np


class GaussianProcessRegressor:

    def __init__(self, kernel, beta):
        """
        construct Gaussian Process Regressor
        :param kernel: kernel function
        :param beta: inverse noise for label t_{n}
        """
        self.kernel = kernel
        self.beta = beta

    def fit(self, X, t, iter_max=None, learning_rate=0.1):
        """
        fit model using gradient ascent (hyperparameters learning)
        :param X: features, (sample_size, n_features)
        :param t: obeserved labels (sample_size, )
        :param iter_max: if set to None, will not learn hyperparameters
        :return: None
        """
        self.X = X
        t = t.reshape(-1)
        self.t = t
        Gram = self.kernel(X, X)
        I = np.eye(X.shape[0])
        self.covariance = Gram + 1 / self.beta * I
        self.precision = np.linalg.inv(self.covariance)
        last_likelihood = -np.Inf
        for i in range(iter_max):
            covariacne_gradients = self.kernel.gradients(X, X)
            # shape: (n_kernel_params, sample_size, sample_size)
            updates = np.array(
                [
                    -0.5 * np.trace(self.precision.dot(grad))
                    + 0.5 * self.t.T.dot(self.precision).dot(grad).dot(self.precision).dot(t)
                    for grad in covariacne_gradients
                ]
            )  # update rule according to (6.70)
            # shape: (n_kernel_params, )
            for j in range(iter_max):
                self.kernel.update_parameters(learning_rate * updates)
                Gram = self.kernel(X, X)
                self.covariance = Gram + 1 / self.beta * I
                self.precision = np.linalg.inv(self.covariance)
                log_likelihood = self._log_likelihood()
                if log_likelihood > last_likelihood:
                    last_likelihood = log_likelihood
                    print("iteration {:d}, log likelihood {:f}".format(i, log_likelihood))
                else:
                    # performance has decreased
                    # undo update
                    # learning rate decay
                    self.kernel.update_parameters(-learning_rate * updates)
                    learning_rate *= 0.9

    def _log_likelihood(self):
        """
        compute log likelihood according to (6.69)
        :return: log likelihood
        """
        return -0.5 * np.linalg.slogdet(self.covariance)[1] \
               - 0.5 * self.t.T.dot(self.precision).dot(self.t) \
               - 0.5 * len(self.t) * np.log(2 * np.pi)

    def predict(self, X, with_error=False):
        """
        make predictions based on (6.66)
        return the mean of the posterior Gaussian distribution
        :param X: (sample_size, n_features)
        :param with_error:
        :return:
        prediction mean: (sample_size, )
        (optional) variance: (sample_size, )
        """
        K = self.kernel(self.X, X)
        # shape: (training_set_size, X_sample_size)
        prediction_mean = K.T.dot(self.precision).dot(self.t)
        if with_error:
            variance = (
                    self.kernel(X, X, pairwise=False)
                    + 1 / self.beta
                    - np.diag(K.T.dot(self.precision).dot(K))
            )
            return prediction_mean, variance

        return prediction_mean
