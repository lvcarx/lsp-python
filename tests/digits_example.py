import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from lsp_python.lsp import LSP

if __name__ == '__main__':
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    X, y = shuffle(X, y, random_state=0)
    X = MinMaxScaler().fit_transform(X)

    to_fit = X[:150]

    lsp = LSP()
    X_init = lsp.fit(to_fit)
    X = lsp.transform(X_init, np.arange(0, 150), X)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=3, cmap='tab10')
    plt.axis('off')
    plt.show()