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
            return 0.0,


class DefaultEvaluator:
    def fit(self, train_X, train_Y):
        self._tX = train_X
        self._trY = train_Y
        self._teX = None
        self._te_Y = None

    def score(self, pipe):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                scores = cross_val_score(pipe, self._tX, self._trY, cv=10)
                return (np.mean(scores),)
        # todo better error handling
        except:
            return 0.0,

    def fit_test(self, test_X, test_Y):
        self._teX = test_X
        self._te_Y = test_Y

    def train_test_score(self, pipe):
        if self._teX is None or self._te_Y is None:
            return None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                pipe.fit(self._tX, self._trY)
                return pipe.score(self._teX, self._te_Y),
        # todo better error handling
        except:
            return 0.0,

