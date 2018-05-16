import matplotlib.pyplot as plt
import numpy as np
from gaussian_process_classifier import GaussianProcessClassifier
from rbf_kernel import RBFKernel


def create_toy_data():
    x0 = np.random.normal(size=50).reshape(-1, 2)
    x1 = np.random.normal(size=50).reshape(-1, 2) + 2.
    return np.concatenate([x0, x1]), np.concatenate([np.zeros(25), np.ones(25)]).astype(np.int)[:, None]


x_train, y_train = create_toy_data()
x0, x1 = np.meshgrid(np.linspace(-4, 6, 100), np.linspace(-4, 6, 100))
x = np.array([x0, x1]).reshape(2, -1).T

model = GaussianProcessClassifier(RBFKernel(np.array([1., 7., 7.])))
model.fit(x_train, y_train, iter_max=100)
y = model.predict(x)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train[:, 0])
plt.contourf(x0, x1, y.reshape(100, 100), levels=np.linspace(0, 1, 3), alpha=0.2)
plt.colorbar()
plt.xlim(-4, 6)
plt.ylim(-4, 6)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("demo/gaussian_process_classifier.pdf")
