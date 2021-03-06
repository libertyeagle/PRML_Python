import numpy as np


class RBFExtendedKernel:

    def __init__(self, params):
        self.params = params
        self.n_params = len(params)
        self.n_features = self.n_params - 3

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
        k(x, y) = params[0] * \exp(-0.5 * \sigma_{i=1}^{i=n_features} params[i + 2]* |x_{i} - y_{i}|^{2})
        + params[1] + parmas[2] * x.T \cdot y
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

        exp_term = -0.5 * np.sum(self.params[3:] * (x - y) ** 2, axis=-1)
        # shape: (sample_size, sample_size) for pairwise

        return self.params[0] * np.exp(exp_term) + self.params[1] + self.params[2] * np.sum(x * y, axis=-1)
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

        grad_param_0 = np.exp(-0.5 * np.sum(self.params[3:] * (x - y) ** 2, axis=-1))
        # shape: (sample_size, sample_size) for pairwise
        grad_param_2 = np.sum(x * y, axis=-1)
        # shape: (sample_size, sample_size)
        grad_params = self.params[0] * grad_param_0[:, :, None] * (-0.5) * (x - y) ** 2
        # shape: (sample_size, sample_size, n_features)

        return np.concatenate(
            (np.expand_dims(grad_param_0, 0),
             np.expand_dims(np.ones(x.shape[:-1]), 0),
             np.expand_dims(grad_param_2, 0),
             grad_params.T)
        )

    def update_params(self, updates):
        """
        update parameters
        :param updates: updates
        :return: None
        """
        self.params += updates
