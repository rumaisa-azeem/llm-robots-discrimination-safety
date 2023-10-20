import numpy as np
from abc import abstractmethod

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.svm import SVC
from choix import ilsr_pairwise
from src.spektrankle_misc import C_to_choix_ls


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


class BradleyTerryRanker(Ranker):

    def __init__(self, C, verbose=True):
        Ranker.__init__(self, self.__class__.__name__, C, verbose)

    def fit(self):
        self.declare()
        choix_ls = C_to_choix_ls(self.C > 0)
        self.r = ilsr_pairwise(self.C.shape[0], choix_ls, alpha=.1)


class Pairwise_LogisticRegression(object):
    """
    Run Pairwise Logistic Regression: By intrinsic design it is
    skew symmetric and has some kind of low rank assumption.
    """

    def __init__(self, choix_ls, y_ls, X):
        super().__init__()
        self.X = X
        self.choix_ls = choix_ls
        self.y_ls = y_ls
        self.log_reg = LogisticRegression()

    def fit(self):
        # Create features
        win_ind = np.array(self.choix_ls)[:, 0]
        loss_ind = np.array(self.choix_ls)[:, 1]
        win_X = np.array(pd.DataFrame(self.X).iloc[win_ind, :])
        loss_X = np.array(pd.DataFrame(self.X).iloc[loss_ind, :])

        # Combine them
        win_pair_X = np.hstack([win_X, loss_X])
        loss_pair_X = np.hstack([loss_X, win_X])
        pair_X = np.vstack([win_pair_X, loss_pair_X])

        # Create outcome
        pair_y = np.array(self.y_ls + [-1 for i in range(len(self.y_ls))])

        # Train the algo
        self.log_reg.fit(pair_X, pair_y)

    def predict(self, choix_ls, predict_proba=True):
        left_ind = np.array(choix_ls)[:, 0]
        right_ind = np.array(choix_ls)[:, 1]

        left_X = np.array(pd.DataFrame(self.X).iloc[left_ind, :])
        right_X = np.array(pd.DataFrame(self.X).iloc[right_ind, :])

        pair_X = np.hstack([left_X, right_X])

        if predict_proba:
            return self.log_reg.predict_proba(pair_X)
        else:
            return self.log_reg.predict(pair_X)

    def predict_unseen_rank(self, X_test, predict_proba=False):

        n = X_test.shape[0]
        choix_ls = C_to_choix_ls(np.ones(shape=(n, n)))

        left_ind = np.array(choix_ls)[:, 0]
        right_ind = np.array(choix_ls)[:, 1]

        left_X = np.array(pd.DataFrame(X_test).iloc[left_ind, :])
        right_X = np.array(pd.DataFrame(X_test).iloc[right_ind, :])

        pair_X = np.hstack([left_X, right_X])

        if predict_proba:
            pred = self.log_reg.predict_proba(pair_X)
        else:
            pred = self.log_reg.predict(pair_X)

        pred = pred.reshape(n, n)
        pred = pred - np.diag(np.diag(pred))

        return pred.mean(axis=1)

    def learning_to_rank(self):
        n = self.X.shape[0]
        full_choix_ls = C_to_choix_ls(np.ones(shape=(n, n)))

        pred = self.predict(full_choix_ls, predict_proba=False).reshape(n, n)
        pred = pred - np.diag(np.diag(pred))

        self.r = pred.mean(axis=1).reshape(-1, 1)


class Pairwise_RandomForest(object):
    """
    Pairwise Random Forest Algorithm
    -
    """

    def __init__(self, choix_ls, y_ls, X):
        super().__init__()
        self.X = X
        self.choix_ls = choix_ls
        self.y_ls = y_ls
        self.rf = RandomForestClassifier()

    def fit(self):
        # Create features
        win_ind = np.array(self.choix_ls)[:, 0]
        loss_ind = np.array(self.choix_ls)[:, 1]
        win_X = np.array(pd.DataFrame(self.X).iloc[win_ind, :])
        loss_X = np.array(pd.DataFrame(self.X).iloc[loss_ind, :])

        # Combine them
        win_pair_X = np.hstack([win_X, loss_X])
        loss_pair_X = np.hstack([loss_X, win_X])
        pair_X = np.vstack([win_pair_X, loss_pair_X])

        # Create outcome
        pair_y = np.array(self.y_ls + [-1 for i in range(len(self.y_ls))])

        # Train the algo
        self.rf.fit(pair_X, pair_y)

    def predict(self, choix_ls, predict_proba=True):
        left_ind = np.array(choix_ls)[:, 0]
        right_ind = np.array(choix_ls)[:, 1]

        left_X = np.array(pd.DataFrame(self.X).iloc[left_ind, :])
        right_X = np.array(pd.DataFrame(self.X).iloc[right_ind, :])

        pair_X_ij = np.hstack([left_X, right_X])
        pair_X_ji = np.hstack([right_X, left_X])

        choix_ls_nega = np.array(choix_ls)[:, [1, 0]]

        # Yij ls
        ij_ls1 = self.rf.predict_proba(pair_X_ij)
        ij_ls2 = self.rf.predict_proba(pair_X_ji)[:, [1, 0]]

        proba_ls = (ij_ls1 + ij_ls2) / 2
        prediction = pd.DataFrame(proba_ls).apply(np.argmax, axis=1) * 2 - 1

        if predict_proba:
            return proba_ls
        else:
            return prediction

    def learning_to_rank(self):
        n = self.X.shape[0]
        full_choix_ls = C_to_choix_ls(np.ones(shape=(n, n)))

        pred = self.predict(full_choix_ls, predict_proba=False).values.reshape(n, n)
        pred = pred - np.diag(np.diag(pred))

        self.r = pred.mean(axis=1)


class Pairwise_GP(object):

    def __init__(self, choix_ls, y_ls, X):
        super().__init__()
        self.X = X
        self.choix_ls = choix_ls
        self.y_ls = y_ls
        self.kernel = kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e1))
        self.gp = GaussianProcessClassifier(kernel=self.kernel)

    def fit(self):
        win_ind = np.array(self.choix_ls)[:, 0]
        loss_ind = np.array(self.choix_ls)[:, 1]
        win_X = np.array(pd.DataFrame(self.X).iloc[win_ind, :])
        loss_X = np.array(pd.DataFrame(self.X).iloc[loss_ind, :])

        # Combine them
        win_pair_X = np.hstack([win_X, loss_X])
        loss_pair_X = np.hstack([loss_X, win_X])
        pair_X = np.vstack([win_pair_X, loss_pair_X])

        # Create outcome
        pair_y = np.array(self.y_ls + [-1 for i in range(len(self.y_ls))])
        print(pair_y)

        # Train the algorithm
        self.gp.fit(pair_X, pair_y)

    def predict(self, choix_ls, predict_proba=True):

        left_ind = np.array(choix_ls)[:, 0]
        right_ind = np.array(choix_ls)[:, 1]

        left_X = np.array(pd.DataFrame(self.X).iloc[left_ind, :])
        right_X = np.array(pd.DataFrame(self.X).iloc[right_ind, :])

        pair_X = np.hstack([left_X, right_X])

        if predict_proba:
            return self.gp.predict_proba(pair_X)
        else:
            return self.gp.predict(pair_X)

    def learning_to_rank(self):
        n = self.X.shape[0]
        full_choix_ls = C_to_choix_ls(np.ones(shape=(n, n)))

        pred = self.predict(full_choix_ls, predict_proba=False).reshape(n, n)
        pred = pred - np.diag(np.diag(pred))

        self.r = pred.mean(axis=1)


class Pairwise_SVM(object):

    def __init__(self, choix_ls, y_ls, X):
        super().__init__()
        self.X = X
        self.choix_ls = choix_ls
        self.y_ls = y_ls
        self.svm = SVC(gamma="auto")

    def fit(self):
        win_ind = np.array(self.choix_ls)[:, 0]
        loss_ind = np.array(self.choix_ls)[:, 1]
        win_X = np.array(pd.DataFrame(self.X).iloc[win_ind, :])
        loss_X = np.array(pd.DataFrame(self.X).iloc[loss_ind, :])

        # Combine them
        win_pair_X = np.hstack([win_X, loss_X])
        loss_pair_X = np.hstack([loss_X, win_X])
        pair_X = np.vstack([win_pair_X, loss_pair_X])

        # Create outcome
        pair_y = np.array(self.y_ls + [-1 for i in range(len(self.y_ls))])
        print(pair_y)

        # Train the algorithm
        self.svm.fit(pair_X, pair_y)

    def predict(self, choix_ls, predict_proba=True):

        left_ind = np.array(choix_ls)[:, 0]
        right_ind = np.array(choix_ls)[:, 1]

        left_X = np.array(pd.DataFrame(self.X).iloc[left_ind, :])
        right_X = np.array(pd.DataFrame(self.X).iloc[right_ind, :])

        pair_X = np.hstack([left_X, right_X])

        if predict_proba:
            return self.svm.predict_proba(pair_X)
        else:
            return self.svm.predict(pair_X)

    def learning_to_rank(self):
        n = self.X.shape[0]
        full_choix_ls = C_to_choix_ls(np.ones(shape=(n, n)))

        pred = self.predict(full_choix_ls, predict_proba=False).reshape(n, n)
        pred = pred - np.diag(np.diag(pred))

        self.r = pred.mean(axis=1)
