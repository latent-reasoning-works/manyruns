"""A toy library class for the call-mode tests. It knows nothing about manyruns."""
import numpy as np


class Toy:
    def __init__(self, k=2):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k

    def fit(self, X):
        self.labels_ = np.arange(len(X)) % self.k
        return self
