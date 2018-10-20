import random
import numpy as np
import multiprocessing

from collections import OrderedDict
import evopipe.deapevo as deapevo
from .evaluator import DefaultEvaluator

from deap import base, creator, tools
from sklearn.pipeline import make_pipeline
from functools import partial


class EvoPipeClassifier:
    def __init__(self, preproc, classif, params, prepro_names, max_prepro=2, ngen=40, pop_size=30, scorer=None,
                 ind_mutpb=0.1, param_mutpb=0.2, swap_mutpb=0.1, len_mutpb=0.4, mutpb=0.25, cxpb=0.5,
                 turns=3, hf_size=5):
        """
        Optimized classification pipeline

        An evolutionary algorithm is used to optimize a scikit-learn pipeline
        :param preproc: Dict of preprocessor names and classes
        :param classif: Dict of classifier names and classes
        :param params: Dictionary of parameter dictionaries, keys are preprocessor/classifier names

        :param max_prepro: Maximum number of preprocessors in the pipeline
        :param ngen: Number of generations
        :param pop_size: Size of a population

        :param ind_mutpb: Probability of a pipeline step to be mutated
        :param param_mutpb: Probability of a single parameter of a pipeline step to be mutated
        :param swap_mutpb: Probability of a whole step to be replaced by another random preprocessor/classifier
                           respectively
        :param len_mutpb: Probability of the individual length to be altered

        :param mutpb: Probability of an individual to be mutated
        :param cxpb: Crossover probability

        :param turns: Tournament selection turns
        :param hf_size: Hall of fame size
        """
        self.prepro_dict = preproc
        self.clf_dict = classif
        self.params_dict = params
        self._scorer = scorer

        self.max_prepro = max_prepro
        self.prepro_names = prepro_names
        self._ngen = ngen
        self._pop_size = pop_size

        self._index_mutpb = ind_mutpb
        self._param_mutpb = param_mutpb
        self._swap_mutpb = swap_mutpb
        self._len_mutpb = len_mutpb
        self._mutpb = mutpb
        self._cxpb = cxpb

        self._trn_size = turns

        self.hof = tools.HallOfFame(hf_size)
        self._toolbox = self._toolbox_init()
        self._eval = DefaultEvaluator()
        self._eval_cache = {}

        self._stats = stats_init()
        self.logbook = tools.Logbook()
        self.logbook.chapters["fitness"].header = "var", "avg", "min", "max"
        self.logbook.chapters["train_test"].header = "var", "avg", "min", "max"
        self.best_pipe = None

    def fit(self, train_X, train_Y, test_X=None, test_Y=None):
        """
        Creates an optimized pipeline and fits it
        :param train_X: Features
        :param train_Y: Target array
        :return: Optimized pipeline
        """

        # evaluator setup
        self._eval.fit(train_X, train_Y)

        if test_X is not None and test_Y is not None:
            self._eval.fit_test(test_X, test_Y)

        # population setup
        pop = self._toolbox.population(n=self._pop_size)

        # population evolution
        pop = deapevo.simple_ea(pop, self._toolbox, self._ngen, self._pop_size, self._cxpb, self._mutpb, self.hof)

        # result hall of fame pipeline
        self.best_pipe = self._compile_pipe(self.hof[0])
        self.best_pipe.fit(train_X, train_Y)
        return self.best_pipe

    def predict(self, test_X):
        """
        Predicts target of test data
        :param test_X: Test features
        :return: Prediction of the target
        """
        if self.best_pipe is None:
            raise NotFittedError("Pipeline is not fitted yet.")

        return self.best_pipe.predict(test_X)

    def score(self, test_X, test_Y):
        """
        Computes prediction score
        :param test_X: Test features
        :param test_Y: Test target
        :return: Prediction score
        """
        if self.best_pipe is None:
            raise NotFittedError("Pipeline is not fitted yet.")

        return self.best_pipe.score(test_X, test_Y)

    def best_pipelines(self):
        """
        Gets a list of best optimized pipelines in the HallOfFame object
        :return: List of optimized pipelines
        """
        pipes = map(self._toolbox.compile, self.hof.items)
        scores = map(lambda ind: ind.fitness.values[0], self.hof.items)
        return zip(pipes, scores)

    # TODO redo
    def _toolbox_init(self):
        """
        Initializes a deap toolbox for the evolutionary algorithm
        :return: Toolbox with registered functios
        """
        toolbox = base.Toolbox()

        creator.create("PipeFitness", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.PipeFitness)

        # pool = multiprocessing.Pool()
        # toolbox.register("map", pool.map)

        toolbox.register("random_prepro", self._random_prepro)
        toolbox.register("random_clf", self._random_clf)

        # individual is a list of tuples (name, parameters)
        toolbox.register("individual", tools.initIterate, creator.Individual, self._ind_range)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        cx_func = partial(deapevo.cx_uniform, prepro_names=self.prepro_names)

        toolbox.register("mate", cx_func)
        toolbox.register("mutate", deapevo.mutate_individual, params=self.params_dict, toolbox=toolbox,
                         prepro_names=self.prepro_names, index_pb=self._index_mutpb, param_pb=self._param_mutpb,
                         swap_pb=self._swap_mutpb, len_pb=self._len_mutpb)

        toolbox.register("evaluate", self._eval_pipe)
        toolbox.register("select", tools.selTournament, tournsize=self._trn_size)

        toolbox.register("log", self._log_stats)
        toolbox.register("compile", self._compile_pipe)

        return toolbox

    def _ind_range(self):
        """
        Creates a random index list based on the preprocessor and classifier arrays and maximum preprocessor number
        :return: List of random indices
        """
        i = 0
        # variable number of preprocessors is allowed
        n_prepro = random.randint(0, self.max_prepro)
        unordered = random.sample(self.prepro_names, n_prepro)
        types = [x for x in self.prepro_names if x in unordered]

        for t in types:
            yield self._random_prepro(t)

        yield self._random_clf()

    def _random_prepro(self, prepro_type, used=None):
        """
        Returns a random preprocessor with parameters chosen randomly from the corresponding params dict
        :return: Name of the preprocessor, parameters
        """
        if prepro_type not in self.prepro_dict.keys():
            raise ValueError("Invalid preprocessor type.")

        # random choice from preprocessor list, corresponding parameters are found in the params dict
        prepro_possible = list(self.prepro_dict[prepro_type].keys())

        name = used
        while name == used:
            name = random.choice(prepro_possible)
            # possibly only one preprocessor (might be equal to used => cycle would not end)
            if len(prepro_possible) == 1:
                break

        return name, self._rand_params(name), prepro_type

    def _random_clf(self):
        """
        Returns a random classifier with parameters chosen randomly from the corresponding params dict
        :return: Name of the classifier, parameters
        """
        # random choice from classifier list, corresponding parameters are found in the params dict
        name = random.choice(list(self.clf_dict.keys()))
        return name, self._rand_params(name)

    def _rand_params(self, name):
        """
        Chooses random parameters from the corresponding dictionary in params
        :param name: Name of the preprocessor or classifier
        :return: A dictionary of randomly set parameters
        """
        # get the corresponding parameter dictionary
        all_params = self.params_dict[name]

        # choose parameters randomly from the lists
        result = OrderedDict()
        for key, values in sorted(all_params.items()):
            rand_param = random.choice(values)
            result[key] = rand_param

        return result

    def _compile_pipe(self, ind):
        """
        Creates a pipeline from the individual
        :param ind: Encoded pipeline, the individual
        :return: Scikit learn pipeline
        """

        # there don't have to be necessarily any preprocessor steps in the pipeline
        if len(ind) > 1:
            # index[0] is preprocessor name, index[1] are preproc. parameters, index[2] is prepro type
            ind_prepro = map(lambda index: self.prepro_dict[index[2]][index[0]](**index[1]), ind[:-1])
        else:
            ind_prepro = []

        ind_clf = map(lambda index: self.clf_dict[index[0]](**index[1]), ind[-1:])

        return make_pipeline(*ind_prepro, *ind_clf)

    def _eval_pipe(self, ind):
        """
        Evaluates the pipeline score
        :param ind: Individual, encrypted pipeline
        :return: Score of the compiled pipeline
        """

        print("\revaluating: {0}".format(ind), end='')
        ind_str = str(ind)
        if ind_str not in self._eval_cache.keys():
            #todo redo
            pipe = self._compile_pipe(ind)

            score = self._eval.score(pipe, scorer=self._scorer)
            tt = self._eval.train_test_score(pipe, scorer=self._scorer)

            self._eval_cache[ind_str] = (score, tt)
            return score, tt

        return self._eval_cache[ind_str]

    def _log_stats(self, pop, gen):
        record = self._stats.compile(pop)
        self.logbook.record(gen=gen, **record)


class NotFittedError(Exception):
    pass


def stats_init():
    fitness_stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    # var_stats = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    train_test_stats = tools.Statistics(key=lambda ind: ind.train_test if ind.train_test is not None else 0.0)

    mstats = tools.MultiStatistics(fitness=fitness_stats, train_test=train_test_stats) # , var_stats=var_stats)

    mstats.register('avg', np.mean)
    mstats.register('var', np.var)
    mstats.register('min', np.min)
    mstats.register('max', np.max)
    return mstats

