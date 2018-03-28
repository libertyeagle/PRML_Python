import numpy as np

def gaussianFeaturesTransform(x, mean, variance, addBias=True):
    """
    :param x:
    (sample_size, ndim) or (sample_size, )
    :param mean:
    (transformed_features_num, ndim) or (transformed_features_num, )
    :param variance:
    int or float
    :return:
    Gaussian Features
    (sample_size, transformed_features_num) or (sample_size, transformed_features_num + 1)
    """
    if mean.ndim == 1:
        mean = mean[:, None]
    else:
        assert mean.ndim == 2
    assert isinstance(variance, int) or isinstance(variance, float)

    if x.ndim == 1:
        x = x[:, None]
    else:
        assert x.ndim == 2
    assert np.size(x, 1) == np.size(mean, 1) # assert each instance x and each feature mean has the same dimension

    # every element in list `transformed` is a row vector of shape (sample_size, )
    # transpose is needed after computation of `transformed`
    if addBias:
        transformed = [np.ones(x.shape[0])]
    else:
        transformed = []
    for each_feature_mean in mean:
        transformed.append(np.exp(-np.sum(np.square(x - mean), axis=1) / (2 * variance)))

    return np.asarray(transformed).transpose()

