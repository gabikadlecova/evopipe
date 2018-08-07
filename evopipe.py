import random
import numpy as np
from evaluator import DefaultEvaluator

from deap import base, creator, tools
from sklearn.pipeline import make_pipeline


class EvoPipeClassifier:
    def __init__(self, preproc, classif, n_preproc, evaluator=None, ngen=20, pop_size=20,
                 ind_mutpb=0.1, mutpb=0.25,cxpb=0.5, turns=3, hf_size=5):
        """
        Optimized classification pipeline

        An evolutionary algorithm is used to optimize a scikit-learn pipeline
        :param preproc: List or ndarray of preprocessors
        :param classif: List or ndarray of classifiers
        :param n_preproc: Maximum number of preprocessors in a pipeline
        :param evaluator: Pipeline evaluator
        :param ngen: Number of generations
        :param pop_size: Size of a population
        :param ind_mutpb: Probability of a pipeline step to be mutated
        :param mutpb: Probability of an individual to be mutated
        :param cxpb: Crossover probability
        :param turns: Tournament selection turns
        :param hf_size: Hall of fame size
        """
        self.pr_l = np.array(preproc)
        self.cl_l = np.array(classif)
        self.n_prep = n_preproc

        self._ngen = ngen
        self._pop_size = pop_size
        self._ind_mutpb = ind_mutpb
        self._mutpb = mutpb
        self._cxpb = cxpb
        self._trn_size = turns

        # todo create a wrapper, evaluator accepts pipelines, not indices or a function
        if evaluator is None:
            self._eval = DefaultEvaluator(self._compile_pipe)
        else:
            self._eval = evaluator

        self.hf = tools.HallOfFame(hf_size)
        self._toolbox = self._toolbox_init()

        self._stats = self._stats_init()
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "avg", "min", "max"
        self.best_pipe = None

    # will be replaced by fit-predict pattern with GridSearch
    def fit(self, train_X, train_Y):
        """
        Creates an optimized pipeline and fits it
        :param train_X: Features
        :param train_Y: Target array
        :return: Optimized pipeline
        """

        self._eval.fit(train_X, train_Y)

        pop = self._toolbox.population(n=self._pop_size)

        fitnesses = map(self._toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(self._ngen):

            offspring = self._toolbox.select(pop, len(pop))
            offspring = list(self._toolbox.map(self._toolbox.clone, offspring))

            for ch1, ch2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self._cxpb:
                    self._toolbox.mate(ch1, ch2)
                    del ch1.fitness.values
                    del ch2.fitness.values

            for mut in offspring:
                if random.random() < self._mutpb:
                    self._toolbox.mutate(mut)
                    del mut.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self._toolbox.evaluate, invalid)

            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            self.hf.update(pop)
            self._log_stats(pop, g)

            if g % 5 == 0:
                print("\nGen {}:\n".format(g + 1))
                # current HallOfFame
                print("Hall of fame:")
                for pipe in self.best_pipelines():
                    pipe_named_steps = []
                    for key, val in pipe.steps:
                        pipe_named_steps.append(key)
                    print(pipe_named_steps)

        self.best_pipe = self._compile_pipe(self.hf[0])
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
        return map(self._toolbox.compile, self.hf.items)

    def _toolbox_init(self):
        """
        Initializes a deap toolbox for the evolutionary algorithm
        :return: Toolbox with registered functios
        """
        toolbox = base.Toolbox()

        creator.create("PipeFitness", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.PipeFitness)

        toolbox.register("individual", tools.initIterate, creator.Individual, self._ind_range)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # -1 indicates "missing preprocessor"
        # the distribution of the probability is not uniform (yet)
        lower_bounds = [-1] * self.n_prep + [0]
        upper_bounds = [len(self.pr_l) - 1] * self.n_prep + [len(self.cl_l) - 1]

        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", tools.mutUniformInt, low=lower_bounds, up=upper_bounds, indpb=self._ind_mutpb)
        toolbox.register("evaluate", self._eval.score)
        toolbox.register("select", tools.selTournament, tournsize=self._trn_size)

        toolbox.register("compile", self._compile_pipe)

        return toolbox

    def _stats_init(self):
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('var', np.var)
        stats.register('min', np.min)
        stats.register('max', np.max)
        return stats

    def _ind_range(self):
        """
        Creates a random index list based on the preprocessor and classifier arrays and maximum preprocessor number
        :return: List of random indices
        """
        i = 0
        while i < self.n_prep:
            i = i + 1
            yield random.randint(0, len(self.pr_l) - 1)

        yield random.randint(0, len(self.cl_l) - 1)

    def _compile_pipe(self, ind):
        """
        Creates a pipeline from the individual
        :param ind: Encoded pipeline, the individual
        :return: Scikit learn pipeline
        """

        valid_prepro = self.n_prep - ind.count(-1)
        valid_inds = [x for x in ind if x != -1]

        prep = self.pr_l[valid_inds[:valid_prepro]]
        clf = self.cl_l[valid_inds[valid_prepro:]]
        return make_pipeline(*prep, *clf)

    def _log_stats(self, pop, gen):
        record = self._stats.compile(pop)
        self.logbook.record(gen=gen, **record)


class NotFittedError(Exception):
    pass