import numpy as np
import itertools
import functools

def polynomialFeaturesTransform(x, degree):
    """
    :param x:
    (sample_size, ndim) or (sample_size, )
    :param degree:
    int
    :return:
    Polynomial Features
    (sample_size, nC0 + nC1 + ... + nC(degree))
    """

    if x.ndim == 1:
        x = x[:, None]
    else:
        assert x.ndim == 2
    
    assert isinstance(degree, int)

    # transpose x in order for itertools to find combinations relative to features, not instances 
    x_transpose = x.transpose()
    transformed = [np.ones(x.shape[0])]
    for i in range(1, degree + 1):
        for item in itertools.combinations_with_replacement(x_transpose, i):
            # add a feature
            transformed.append(functools.reduce(lambda x, y: x * y, item))
    
    return np.asarray(transformed).transpose()

