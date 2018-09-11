import numpy as np
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


class TrainTestEvaluator:

    def __init__(self, preproc, classif):
        self.pr = np.array(preproc)
        self.cl = np.array(classif)
        self.tr_x = None
        self.tr_y = None
        self.te_x = None
        self.te_y = None

    def fit(self, train_X, train_Y):
        self.tr_x = train_X
        self.tr_y = train_Y

    def score(self, individual, n_prepro, test_X, test_Y):
        prep = self.pr[individual[:n_prepro]]
        clf = self.cl[individual[n_prepro:]]
        pipe = make_pipeline(*prep, *clf)
        try:
            pipe.fit(self.tr_x, self.tr_y)
            return (pipe.score(test_X, test_Y),)
        # todo better error handling
        except:
            return (0.0,)


class DefaultEvaluator:
    def fit(self, train_X, train_Y):
        self._tX = train_X
        self._tY = train_Y

    def score(self, pipe):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                scores = cross_val_score(pipe, self._tX, self._tY)
                return (np.mean(scores),)
        # todo better error handling
        except:
            return (0.0,)
