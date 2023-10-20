import torch
from torch.nn import Sigmoid
from gpytorch.kernels import RBFKernel
from src.spektrankle_misc import median_heuristic, binarise
from torch.optim import Adam


class PrefKRR(torch.nn.Module):

    def __init__(self, n, d, lengthscale):
        super().__init__()
        self.kernel = RBFKernel(ard_num_dims=d)
        self.kernel.lengthscale = lengthscale
        self.kernel.raw_lengthscale.requires_grad = False
        self.alphas = torch.nn.Parameter(torch.randn((n, 1))).float()

    def forward(self, X):
        self.K = self.kernel(X)
        f = self.K.matmul(self.alphas)
        ones = torch.ones_like(f)
        f_diff = f @ ones.T - ones @ f.T

        self.f = f
        return f_diff


class PreferentialKRR(object):

    def __init__(self, C, X, verbose=False):

        # Change C to +1 and -1
        self.C = torch.tensor(binarise(C)).float()
        self.X = torch.tensor(X).float()
        self.krr = PrefKRR(n=X.shape[0], d=X.shape[1], lengthscale=median_heuristic(X))
        self.sig = Sigmoid()
        self.verbose = verbose

    def fit(self, epoch=1000, lr=1e-1, reg=0):
        optim = Adam(self.krr.parameters(), lr=lr)

        for rd in range(epoch):
            f_diff = self.krr(self.X)
            loss = -torch.sum(torch.log(
                self.sig(f_diff * self.C))) + reg * self.krr.alphas.T @ self.krr.K.evaluate() @ self.krr.alphas

            if self.verbose:
                if rd % 100 == 0:
                    print(loss)

            loss.backward()
            optim.step()
            optim.zero_grad()

        self.f = self.krr.f.detach().numpy()

    def predict(self, K_test):

        return K_test @ self.krr.alphas.detach().numpy()
