from abc import abstractmethod

import numpy as np
from numpy import shape, reshape
from numpy import zeros, mean, around, \
    dot, argsort, allclose, fill_diagonal, abs, matrix
from numpy.linalg import solve, eig, eigh, cholesky
from scipy.linalg import eig
from sklearn.cross_decomposition import CCA

import src.rcca as rcca
from src.spektrankle_misc import Compute_Sim, GraphLaplacian, binarise, median_heuristic
from src.spektrankle_misc import centering_matrix, get_the_subspace_basis, \
    compute_upsets, invsqrtm_psd, extract_upsets


class Ranker(object):
    """
    Set up the Ranker Class type for our algorithms
    """

    def __init__(self, name, C, verbose=True):
        self.name = name
        self.C = C
        self.verbose = verbose

    def declare(self):
        if (self.verbose):
            print("\nRunning " + self.name)

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError


class SVDRankerNormal(Ranker):
    """
    Implementation of the SVD-Norm Rank algorithm
    """

    def __init__(self, C, verbose=True):
        """
        Inputs:
            - C: Pairwise comparison matrix
        """
        Ranker.__init__(self, self.__class__.__name__, C, verbose)

    def fit(self):
        self.declare()
        n = shape(self.C)[0]  # number of items
        D_inv = zeros((n, n))
        diag_ls = matrix(abs(self.C)).sum(axis=1)
        diag_inv = [1 / i if i != 0 else 0 for i in diag_ls]
        fill_diagonal(D_inv, diag_inv)
        self.C = dot(D_inv, self.C)
        # no side - eigendecompose C*C.T
        w, V = eigh(dot(self.C, (self.C).T))
        ind = argsort(-w)  # order eigenvalues descending
        w = w[ind]
        V = V[:, ind]  # second axis !!
        if (self.verbose):
            print("top 10 eigenvalues:", around(w[:10], decimals=2))

        # use the number of upsets to select between eigenvectors and signs
        r0 = reshape(V[:, 0], (n, 1))
        r1 = reshape(V[:, 1], (n, 1))
        u0, m0, r0sign = compute_upsets(
            r0, self.C, which_method=self.name + "1st eig",
            verbose=self.verbose)
        u1, m1, r1sign = compute_upsets(
            r1, self.C, which_method=self.name + "2nd eig",
            verbose=self.verbose)
        if (r0sign == -1):
            u0 = m0
        if (r1sign == -1):
            u1 = m1
        if (u1 < u0):
            # print("second!")
            self.r = r1sign * r1
        else:
            # print("first!")
            self.r = r0sign * r0


class SVDRankerCov(Ranker):
    """
    Implement the SVDCov-Rank algorithm
    """

    def __init__(self, C, Phi, verbose=True, chol=False):
        """
        Inputs:
            - C: Pairwise comparison matrix
            - Phi: Feature matrix
        """
        Ranker.__init__(self, self.__class__.__name__, C, verbose)
        self.Phi = Phi
        self.chol = chol

    def fit(self):
        self.declare()
        n = shape(self.C)[0]
        H = centering_matrix(n)
        # feature-space-covariance
        PhCov = dot((self.Phi).T, dot(H, self.Phi))
        # feature-match cross-covariance
        PhHC = dot((self.Phi).T, dot(H, self.C))
        if (self.chol):
            L = cholesky(PhCov)
            if (self.verbose):
                print("cholesky check: ", allclose(dot(L, L.T), PhCov))
            LiPhHC = solve(L, PhHC)
        else:
            Linv = invsqrtm_psd(PhCov)
            LiPhHC = Linv.dot(PhHC)
        M = dot(LiPhHC, LiPhHC.T)
        self.simG = M
        w, V = eigh(M)
        ind = argsort(-w)  # order eigenvalues descending
        w = w[ind]
        V = V[:, ind]  # second axis !!
        if (self.verbose):
            print("top 10 eigenvalues:", around(w[:10], decimals=2))
        if (self.chol):
            beta = solve(L.T, V[:, 0])
        else:
            beta = Linv.dot(V[:, 0])
        rcov = dot(self.Phi, beta)
        rcov = rcov - mean(rcov)  # H*Phi*beta
        rcov = reshape(rcov, (n, 1))
        # choose sign using the number of upsets
        _, _, rsign = compute_upsets(rcov, self.C, verbose=False)
        self.r = rsign * rcov
        self.beta = rsign * beta

    def predict(self, Phi_):
        return Phi_ @ self.beta


class SVDRankerNCov(Ranker):
    """
    Implement an alternative of SVDCov-Rank based on SVDNorm-Rank setup on the
    normalised C matrix
    """

    def __init__(self, C, Phi, verbose=True, chol=False):
        """
        Inputs:
            - C: Pairwise comparison matrix
            - Phi: Feature matrix
        """
        Ranker.__init__(self, self.__class__.__name__, C, verbose)
        self.Phi = Phi
        self.chol = chol

    def fit(self):
        self.declare()
        n = shape(self.C)[0]
        D_inv = zeros((n, n))
        diag_ls = matrix(abs(self.C)).sum(axis=1)
        diag_inv = [1 / i if i != 0 else 0 for i in diag_ls]
        fill_diagonal(D_inv, diag_inv)
        self.C = dot(D_inv, self.C)
        H = centering_matrix(n)
        # feature-space-covariance
        PhCov = dot((self.Phi).T, dot(H, self.Phi))
        # feature-match cross-covariance
        PhHC = dot((self.Phi).T, dot(H, self.C))
        if (self.chol):
            L = cholesky(PhCov)
            if (self.verbose):
                print("cholesky check: ", allclose(dot(L, L.T), PhCov))
            LiPhHC = solve(L, PhHC)
        else:
            Linv = invsqrtm_psd(PhCov)
            LiPhHC = Linv.dot(PhHC)
        M = dot(LiPhHC, LiPhHC.T)
        self.simG = M
        w, V = eigh(M)
        ind = argsort(-w)  # order eigenvalues descending
        w = w[ind]
        V = V[:, ind]  # second axis !!
        if (self.verbose):
            print("top 10 eigenvalues:", around(w[:10], decimals=2))
        if (self.chol):
            beta = solve(L.T, V[:, 0])
        else:
            beta = Linv.dot(V[:, 0])
        rcov = dot(self.Phi, beta)
        rcov = rcov - mean(rcov)  # H*Phi*beta
        rcov = reshape(rcov, (n, 1))
        _, _, rsign = compute_upsets(rcov, self.C, verbose=False)
        self.r = rsign * rcov
        self.beta = rsign * beta

    def predict(self, Phi_):
        return Phi_ @ self.beta


class SVDRankerKCov(Ranker):
    """
    Kernelised version of SVDCov-Rank
    """

    def __init__(self, C, K, verbose=True):
        """
        Inputs:
            - C: pairwise compairons Matrix
            - K: Kernel matrix
        """
        Ranker.__init__(self, self.__class__.__name__, C, verbose)
        self.K = K

    def fit(self):
        self.declare()
        n = shape(self.C)[0]
        H = centering_matrix(n)
        # feature-space-covariance
        KHK = dot(self.K, dot(H, self.K))
        # feature-match cross-covariance
        KHC = dot(self.K, dot(H, self.C))
        Linv = invsqrtm_psd(KHK)
        LiKHC = Linv.dot(KHC)
        M = dot(LiKHC, LiKHC.T)
        self.simG = M
        w, V = eigh(M)
        ind = argsort(-w)  # order eigenvalues descending
        w = w[ind]
        V = V[:, ind]  # second axis !!
        alpha = Linv.dot(V[:, 0])
        r = dot(self.K, alpha)
        r = r - mean(r)
        r = reshape(r, (n, 1))
        # choose sign using the number of upsets
        _, _, rsign = compute_upsets(
            r, self.C, which_method=self.name, verbose=self.verbose)
        self.r = rsign * r
        self.alpha = rsign * alpha

    def predict(self, K_test):
        return K_test @ self.alpha


class SVDRankerNKCov(Ranker):
    """
    Kernelised version of the SVDRankerNCov
    """

    def __init__(self, C, K, verbose=True):
        """
        Inputs:
            - C: Pairwise Comparison Matrix
            - K: Kernel matrix
        """
        Ranker.__init__(self, self.__class__.__name__, C, verbose)
        self.K = K

    def fit(self):
        self.declare()
        n = shape(self.C)[0]
        D_inv = zeros((n, n))
        diag_ls = matrix(abs(self.C)).sum(axis=1)
        diag_inv = [1 / i if i != 0 else 0 for i in diag_ls]
        fill_diagonal(D_inv, diag_inv)
        self.C = dot(D_inv, self.C)
        H = centering_matrix(n)
        # feature-space-covariance
        KHK = dot(self.K, dot(H, self.K))
        # feature-match cross-covariance
        KHC = dot(self.K, dot(H, self.C))
        Linv = invsqrtm_psd(KHK)
        LiKHC = Linv.dot(KHC)
        M = dot(LiKHC, LiKHC.T)
        self.simG = M
        w, V = eigh(M)
        ind = argsort(-w)  # order eigenvalues descending
        w = w[ind]
        V = V[:, ind]  # second axis !!
        alpha = Linv.dot(V[:, 0])
        r = dot(self.K, alpha)
        r = r - mean(r)
        r = reshape(r, (n, 1))
        # choose sign using the number of upsets
        _, _, rsign = compute_upsets(
            r, self.C, which_method=self.name, verbose=self.verbose)
        self.r = rsign * r
        self.alpha = rsign * alpha

    def predict(self, K_test):
        return K_test @ self.alpha


# ------------------------
##########################
# Seriation Based Method #
##########################


class SerialRank(Ranker):
    """
    A class that runs the Serial Rank algorithm

    The C matrix is a binarised comparison matrix.
    """

    def __init__(self, C, verbose=True):
        """
        Inputs:
            - C: pairwise comparison matrix
        """
        Ranker.__init__(self, self.__class__.__name__, C, verbose)

    def fit(self):
        # First compute similarity matrix
        S = Compute_Sim(self.C)
        n = S.shape[0]
        Z = get_the_subspace_basis(n, verbose=False)

        Ls = GraphLaplacian(S)
        ztLsz = dot(dot(Z.T, Ls), Z)
        w, v = eig(ztLsz)

        ind = np.argsort(w)
        v = v[:, ind]
        r = np.reshape(Z.dot(v[:, 0]), (n, 1))

        _, _, rsign = compute_upsets(r, S, verbose=False)

        self.r = rsign * r


class CSerialRank(Ranker):
    """
    A class that runs constrained serial ranking I

    The C matrix is a binarised comparison matrix
    and side information is given as a kernel matrix
    """

    def __init__(self, C, K, ita, verbose=True):
        """
        Inputs:
            - C: Pairwise comparison matrix
            - K: Kernel Matrix
            - ita: Regularisation parameter
        """
        Ranker.__init__(self, self.__class__.__name__, C, verbose)
        self.K = K
        self.ita = ita

    def fit(self):
        # First compute similarity matrix
        S = Compute_Sim(self.C)
        n = S.shape[0]
        Z = get_the_subspace_basis(n, verbose=False)

        Ls = GraphLaplacian(S)
        Lk = GraphLaplacian(self.K)
        ztLspluslLkz = dot(dot(Z.T, (Ls + self.ita * Lk)), Z)
        w, v = eig(ztLspluslLkz)

        ind = np.argsort(w)
        v = v[:, ind]
        r = np.reshape(Z.dot(v[:, 0]), (n, 1))

        _, _, rsign = compute_upsets(r, self.C, verbose=False)

        self.r = rsign * r


# ------------------------
#####################
# CCA Based method  #
#####################


class CCARank(Ranker):
    """
    A class that runs the CC Rank Algorithm
    """

    def __init__(self, C, X, verbose=True):
        """
        Inputs:
            - C: pairwise comparison matrix
        """
        Ranker.__init__(self, self.__class__.__name__, C, verbose)
        self.X = X

    def fit(self):
        # First compute similarity matrix
        S = Compute_Sim(self.C)
        cca = CCA(n_components=1)
        cca.fit(self.X, S)
        self.cca = cca

        _, _, rsign = compute_upsets(cca.x_scores_, S, verbose=False)
        self.r = rsign * cca.x_scores_

    def predict(self, X):
        # normalise the vector
        return (self.cca.predict(X)[:, 0] - self.cca.predict(X)[:, 0].mean()) / self.cca.predict(X)[:, 0].std()


def _make_kernel(d, normalize=False, ktype='linear', gausigma=1.0, degree=2):
    """Makes a kernel for data d
      If ktype is 'linear', the kernel is a linear inner product
      If ktype is 'gaussian', the kernel is a Gaussian kernel, sigma = gausigma
      If ktype is 'poly', the kernel is a polynomial kernel with degree=degree
      If ktype is 'rbf', the kernel is a RBF kerenl with sigma = gausigma
    """
    d = np.nan_to_num(d)
    cd = _demean(d)
    if ktype == 'linear':
        kernel = np.dot(cd, cd.T)
    elif ktype == 'gaussian':
        from scipy.spatial.distance import pdist, squareform
        pairwise_dists = squareform(pdist(d, 'euclidean'))
        kernel = np.exp(-pairwise_dists ** 2 / 2 * gausigma ** 2)
    elif ktype == 'poly':
        kernel = np.dot(cd, cd.T) ** degree
    kernel = (kernel + kernel.T) / 2.
    if normalize:
        kernel = kernel / np.linalg.eigvalsh(kernel).max()
    return kernel


def _demean(d): return d - d.mean(0)


class KCCARank(Ranker):
    """
    Implementation of the KCCA-Rank algorithm
    """

    def __init__(self, C, Phi, reg, verbose=True, lengthscale=1.0):
        """
        Inputs:
            - C: The pairwise comparison matrix
            - Phi: The feature matrix
            - reg: The strength of regularisation for the kcca algorithm
        """
        Ranker.__init__(self, self.__class__.__name__, C, verbose)
        self.md = median_heuristic(Phi, type="nontorch")
        self.Phi = Phi / self.md
        self.reg = reg
        self.lengthscale = lengthscale

    def fit(self):

        # S = Compute_Sim(self.C)/self.C.shape[0]
        # S = S / median_heuristic(S, type="nontroch")

        cca = rcca.CCA(kernelcca=True, reg=self.reg, numCC=10,
                       verbose=self.verbose, ktype="gaussian", lengthscale=self.lengthscale)
        cca.train([self.Phi, self.C @ self.C.T])
        # cca.train([self.Phi, S])

        W = cca.comps[1]
        V = cca.comps[0]
        ind = np.argmax(cca.cancorrs)
        _, _, rsign = compute_upsets(W[:, ind], self.C, verbose=False)

        self.r = rsign * W[:, ind]
        self.rII = rsign * V[:, ind]
        if extract_upsets(self.r, self.C) < extract_upsets(self.rII, self.C):
            self.r = self.rII
        else:
            self.r = self.r

        self.cca = cca

        self.K = _make_kernel(self.Phi, ktype="gaussian")
        try:
            K_inv = np.linalg.inv(self.K + 1e-2 * np.eye(self.K.shape[0]))
        except:
            K_inv = np.linalg.inv(self.K + 1e-1 * np.eye(self.K.shape[0]))
        self.alpha = K_inv @ self.r

    def predict(self, Phi_):
        """
        New features
        """
        n_player_old = self.Phi.shape[0]
        Phi_ = Phi_ / self.md
        full_Phi = np.concatenate([self.Phi, Phi_], axis=0)

        K_new = _make_kernel(full_Phi, ktype="gaussian")
        K_input = K_new[n_player_old:, :n_player_old]

        return K_input @ self.alpha


########################
# Diffusion Centrality #
########################

#
# class DiffusionRankCentrality(Ranker):
#
#     def __init__(self, C, K, verbose=False):
#         Ranker.__init__(self, self.__class__.__name__, binarise(C), verbose)
#         self.K = K
#         self.m = (binarise(C) > 0).sum()
#         self.C_pre = C
#
#     def fit(self):
#
#         C = (self.C < 0)/self.m
#         C += np.diag(1 - C.sum(axis=1))
#         K = self.K/self.K.sum(axis=1)
#         Q = C @ self.K
#         e_val, e_vec = eig(Q, left=True, right=False)
#         r1 = e_vec[:, 0].real
#         r2 = e_vec[:, -1].real
#
#         if extract_upsets(r1, self.C_pre) < extract_upsets(r2, self.C_pre):
#             self.r = r2
#         else:
#             self.r = r1

class DiffusionRankCentrality(Ranker):

    def __init__(self, C, K, verbose=False):
        Ranker.__init__(self, self.__class__.__name__, binarise(C), verbose)
        self.K = K / K.sum(axis=1)
        self.m = (binarise(C) > 0).sum()
        self.C_pre = C

    def fit(self):
        S = Compute_Sim(self.C) / self.C.shape[0]
        Q = (self.C > 0) * (1 - S / 2) + (self.C < 0) * (S / 2)
        Q_ = Q @ self.K
        e_val, e_vec = eig(Q_, left=True, right=False)
        r = e_vec[:, 0].real

        self.r = r


class RankCentrality(Ranker):

    def __init__(self, C, verbose=False):
        Ranker.__init__(self, self.__class__.__name__, binarise(C), verbose)
        self.m = (binarise(C) > 0).sum()
        self.C_pre = C

    def fit(self):
        S = Compute_Sim(self.C) / self.C.shape[0]
        Q = (self.C > 0) * (1 - S / 2) + (self.C < 0) * (S / 2)
        Q_ = Q
        e_val, e_vec = eig(Q_, left=True, right=False)
        r = e_vec[:, 0].real

        self.r = r


# ------------------------
#################
# SVDKFair-Rank #
#################


class SVDRankerFair(Ranker):
    """
    Implementation of the SVDKFair algorithm
    """

    def __init__(self, C, K, W, lmda, verbose=True):
        """
        Inputs:
         - C: The skew symmetric pairwise comparison matrix
         - K: The kernel matrix for your predictors
         - W: The kernel matrix for your sensitive variables
         - lmda: The regularisation parameter
        """
        Ranker.__init__(self, self.__class__.__name__, C, verbose)
        self.K = K
        self.W = W
        self.lmda = lmda

    def fit(self):
        self.declare()
        n = shape(self.C)[0]
        D_inv = zeros((n, n))
        diag_ls = matrix(abs(self.C)).sum(axis=1)
        diag_inv = [1 / i if i != 0 else 0 for i in diag_ls]
        fill_diagonal(D_inv, diag_inv)
        self.C = dot(D_inv, self.C)
        H = centering_matrix(n)

        # SVDKCov Ranking objective
        KHK = dot(self.K, dot(H, self.K))
        KHC = dot(self.K, dot(H, self.C))
        Linv = invsqrtm_psd(KHK)
        LiKHC = Linv.dot(KHC)
        M1 = dot(LiKHC, LiKHC.T)

        # HSIC Regularisation

        LinvKH = dot(dot(Linv, self.K), H)
        M2 = dot(dot(LinvKH, self.W), LinvKH.T)

        # Optimisation
        M3 = M1 - self.lmda * n ** (-2) * M2

        w, V = eigh(M3)
        ind = argsort(-w)  # order eigenvalues descending
        w = w[ind]
        V = V[:, ind]  # second axis !!
        alpha = Linv.dot(V[:, 0])
        r = dot(self.K, alpha)
        r = r - mean(r)
        r = reshape(r, (n, 1))
        # choose sign using the number of upsets
        _, _, rsign = compute_upsets(
            r, self.C, which_method=self.name, verbose=self.verbose)
        self.r = rsign * r
        self.alpha = rsign * alpha

        self.HSIC = n ** (-2) * dot(V[:, 0].T,
                                    dot(self.lmda * n ** (-2) * M2, V[:, 0]))

    def compute_HSIC(self, f):
        n = shape(self.C)[0]
        H = centering_matrix(n)

        HWH = dot(dot(H, self.W), H)
        HSIC = n ** (-2) * dot(dot(f.T, HWH), f)
        if self.verbose:
            print(HSIC)
        return HSIC
