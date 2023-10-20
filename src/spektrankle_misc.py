import copy

import numpy as np
import torch
from numpy import sign, count_nonzero, ones, shape, reshape, eye, \
    dot, sqrt, argsort, allclose, concatenate, empty, arange
from numpy.linalg import eigh
from scipy.stats import kendalltau
from sklearn.metrics import pairwise_distances


def kendalltau_score(r1, r2):
    return np.abs(kendalltau(r1, r2)[0])


def sqrtm_psd(A, check_finite=True):
    """
    Returns the matrix square root of a positive semidefinite matrix,
    truncating negative eigenvalues.
    """
    s, V = eigh(A)
    s[s <= 0] = 0
    s = sqrt(s)
    A_sqrt = (V * s).dot(V.conj().T)
    return A_sqrt


def invsqrtm_psd(A, check_finite=True):
    """
    Returns the inverse matrix square root of a positive semidefinite matrix,
    truncating negative eigenvalues.
    """
    s, V = eigh(A)
    s[s <= 1e-16] = 0
    s[s > 0] = 1 / sqrt(s[s > 0])
    A_invsqrt = (V * s).dot(V.conj().T)
    return A_invsqrt


def centering_matrix(n):
    # centering matrix, projection to the subspace orthogonal
    # to all-ones vector
    return eye(n) - ones((n, n)) / n


def rank_items(array):
    temp = array.argsort()
    ranks = empty(len(array), int)
    ranks[temp] = arange(len(array))
    return ranks + 1


def get_the_subspace_basis(n, verbose=True):
    # returns the orthonormal basis of the subspace orthogonal
    # to all-ones vector
    H = centering_matrix(n)
    s, Zp = eigh(H)
    ind = argsort(-s)  # order eigenvalues descending
    s = s[ind]
    Zp = Zp[:, ind]  # second axis !!
    if (verbose):
        print("...forming the Z-basis")
        print("check eigenvalues: ", allclose(
            s, concatenate((ones(n - 1), [0]), 0)))

    Z = Zp[:, :(n - 1)]
    if (verbose):
        print("check ZZ'=H: ", allclose(dot(Z, Z.T), H))
        print("check Z'Z=I: ", allclose(dot(Z.T, Z), eye(n - 1)))
    return Z


def compute_upsets(r, C, verbose=True, which_method=""):
    n = shape(r)[0]
    totmatches = count_nonzero(C) / 2
    if (len(shape(r)) == 1):
        r = reshape(r, (n, 1))
    e = ones((n, 1))
    Chat = r.dot(e.T) - e.dot(r.T)
    upsetsplus = count_nonzero(sign(Chat[C != 0]) != sign(C[C != 0]))
    upsetsminus = count_nonzero(sign(-Chat[C != 0]) != sign(C[C != 0]))
    winsign = 2 * (upsetsplus < upsetsminus) - 1
    if (verbose):
        print(which_method + " upsets(+): %.4f" %
              (upsetsplus / float(2 * totmatches)))
        print(which_method + " upsets(-): %.4f" %
              (upsetsminus / float(2 * totmatches)))
    return upsetsplus / float(2 * totmatches), upsetsminus / float(2 * totmatches), winsign


def Compute_Sim(C):
    """
    Compute the Similarity matrix
    """
    n = C.shape[0]
    ones_mat = n * np.dot(np.ones(n).reshape(-1, 1), np.ones(n).reshape(1, -1))
    S = 0.5 * (ones_mat + np.dot(C, C.T))
    return S


def GraphLaplacian(G):
    """
    Input a simlarity graph G and return graph GraphLaplacian
    """
    D = np.diag(G.sum(axis=1))
    L = D - G

    return L


def Signed_GraphLaplacian(G):
    """
    Compute the signed graph laplacian of a signed graph
    """

    D = np.diag(np.abs(G).sum(axis=1))
    L = D - G

    return L


def Separate_Constraints(Q):
    """
    Separate the constraints of the given information
    Return: a matrix of positive constraints that you want to encourage
            another matrix of negative constraints that you want to discourage
    """
    Q_plus = copy.deepcopy(Q)
    Q_minus = copy.deepcopy(Q)
    Q_plus[Q < 0] = 0
    Q_minus[Q > 0] = 0

    return Q_plus, -1 * Q_minus


def median_heuristic(X, type="torch"):
    if X.shape[1] == 1:
        lengthscales = np.median(pairwise_distances(X))
    else:
        lengthscales = []
        for j in range(X.shape[1]):
            lengthscales.append(np.median(pairwise_distances(X[:, [j]])))

        for i, val in enumerate(lengthscales):
            if val == 0:
                lengthscales[i] = 1

    if type == "torch":
        return torch.tensor(lengthscales)
    else:
        return lengthscales


def C_to_choix_ls(C):
    win_matches = np.nonzero(C > 0)
    return np.concatenate((win_matches[0].reshape(-1, 1), win_matches[1].reshape(-1, 1)), axis=1)


def train_test_split_C(C, train_ratio=0.7):
    C_train, C_test = np.zeros_like(C), np.zeros_like(C)
    cut = round(C.shape[0] * train_ratio)
    C_train[:cut, :cut] = C[:cut, :cut]
    C_test[cut:, cut:] = C[cut:, cut:]

    return C_train, C_test


def train_test_split_C_random(C, X, train_ratio=0.7):
    shuffle_ind = np.random.choice(range(C.shape[0]), replace=False, size=C.shape[0])
    X = X[shuffle_ind, :]
    C = C[shuffle_ind, :][:, shuffle_ind]

    C_train, C_test = np.zeros_like(C), np.zeros_like(C)
    cut = round(C.shape[0] * train_ratio)
    C_train[:cut, :cut] = C[:cut, :cut]
    C_test[cut:, cut:] = C[cut:, cut:]

    return C_train, C_test, X


def binarise(C):
    return np.sign(C)


def choix_ls_to_C(ls, n):
    C = np.zeros((n, n))
    for rd, match in enumerate(ls):
        i, j = match[0], match[1]

        C[i, j] = 1
        C[j, i] = -1

    return C


def extract_upsets(r, C):
    a, b, _ = compute_upsets(r, C, verbose=False)
    return max([a, b])


def train_test_split_random_seen_players(C, X, split=0.5):
    choix_ls = C_to_choix_ls(C)
    shuffle_ind = np.random.choice(len(choix_ls), replace=False, size=len(choix_ls))
    choix_ls = choix_ls[shuffle_ind]
    cut = int(len(choix_ls) * split)
    train_ls, test_ls = choix_ls[: cut], choix_ls[cut:]

    C_train = choix_ls_to_C(train_ls, X.shape[0])
    C_test = choix_ls_to_C(test_ls, X.shape[0])

    return C_train, C_test
