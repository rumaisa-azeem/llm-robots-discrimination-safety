"""
Simulation Script

"""

import numpy as np


def generate_match(r, sparsity_rate=0.9):
    n = len(r)
    ones = np.ones_like(r)
    C = r @ ones.T - ones @ r.T

    mask = np.random.choice([0, 1], size=(n, n), replace=True, p=[1 - sparsity_rate, sparsity_rate])
    mask = np.triu(mask) + np.triu(mask).T

    return C * mask


class Simulation(object):

    def __init__(self, n, d, sparsity_rate, p, type="FLIP"):
        """
        Running the simulation model. Either FLIP or ERO.

        :param n: number of players
        :param d: number of dimension of features
        :param sparsity_rate: sparsity rate
        :param p: parameter for the error model
        :param type:
        """

        self.n = n
        self.d = d
        self.sparsity_rate = sparsity_rate
        # probability of flip
        self.p = p

        self.type = type

        # generate players
        self.X = np.random.random(size=(self.n, self.d))
        self.beta = np.random.random(size=(self.d, 1))
        if d == 1:
            self.projection = self.X @ self.beta
        else:
            self.projection = self.X

    def generate(self, noise):
        # generate skill function
        r = np.sin(3 * np.pi * self.projection - 1.5 * self.projection ** 2) + noise * np.random.normal(
            size=(self.X.shape[0], 1))
        self.true_r = r
        ones = np.ones_like(r)

        C = r @ ones.T - ones @ r.T
        self.C_full = C

        if self.type == "FLIP":
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if np.random.uniform() <= self.p:
                        C[i, j], C[j, i] = C[j, i], C[i, j]

        elif self.type == "ERO":
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if np.random.uniform() <= self.p:
                        replacement = C[i, j] * (np.random.uniform() * 2 - 1)
                        C[i, j], C[j, i] = replacement, -replacement
        self.C_corrupt = C

        # Inject sparsity
        mask = np.random.choice([0, 1],
                                size=(self.n, self.n),
                                replace=True,
                                p=[self.sparsity_rate, 1 - self.sparsity_rate]
                                )
        mask = np.triu(mask) + np.triu(mask).T
        self.C = C * mask
