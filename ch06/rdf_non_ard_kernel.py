import numpy as np


class RBFKernel:

    def __init__(self, params, n_features):
        self.params = params
        self.n_features = n_features

    def _pairwise(self, x, y):
        """
        return two matrix contains all possible pairs for x and y
        :param x: (sample_size, n_features)
        :param y: (sample_size, n_features)
        :return:
        a tuple contains two matrix of shape: (sample_size, sample_size, n_features)
        the elements of i_th row of the first matrix are all sample x_(i)
        the elements of j_th column of the second matrix are all sample y_(j)
        Gram matrix can be calculated based on this function
        """

        return (
            np.tile(x, (len(y), 1, 1)).transpose((1, 0, 2)),
            np.tile(y, (len(x), 1, 1))
        )

    def __call__(self, x, y, pairwise=True):
        """
        compute kernel function
        k(x, y) = params[0] * \exp(-0.5 * params[1] * ||x - y||^{2})
        :param x: (sample_size, n_features)
        :param y: (sample_size, n_features)
        :param pairwise: compute kernel function for every pair of (x, y) or just for matching terms k(x_(i), x_(i))
        :return:
        (sample_size, sample_size)
        or
        (sample_size, )
        """
        if pairwise == True:
            # for multiple samples at the same time
            x, y = self._pairwise(x, y)

        exp_term = -0.5 * self.params[1] * np.sum((x - y) ** 2, axis=-1)
        # shape: (sample_size, sample_size) for pairwise

        return self.params[0] * np.exp(exp_term)
        # shape: (sample_size, sample_size) for pairwise

    def gradients(self, x, y, pairwise=True):
        """
        compute gradients of kernel function
        :param x: (sample_size, n_features)
        :param y: (sample_size, n_features)
        :param pairwise: compute kernel function for every pair of (x, y) or just for matching terms k(x_(i), x_(i))
        :return: (n_params, sample_size, sample_size)
        return gradient for each parameter at each data point
        """
        if pairwise == True:
            x, y = self._pairwise(x, y)

        grad_param_0 = np.exp(-0.5 * self.params[1] * np.sum((x - y) ** 2, axis=-1))
        # shape: (sample_size, sample_size) for pairwise
        grad_param_1 = self.params[0] * grad_param_0 * (-0.5) * np.sum((x - y) ** 2, axis=-1)
        # shape: (sample_size, sample_size)

        return np.concatenate(
            (np.expand_dims(grad_param_0, 0),
             np.expand_dims(grad_param_1, 0))
        )

    def update_params(self, updates):
        """
        update parameters
        :param updates: updates
        :return: None
        """
        self.params += updates
