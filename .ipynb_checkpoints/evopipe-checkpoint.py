import random
import numpy as np
from evaluator import DefaultEvaluator

from deap import base, creator, tools
from functools import partial
from sklearn.pipeline import make_pipeline


class EvoPipeClassifier:
    def __init__(self, preproc, classif, n_preproc, evaluator=None, ngen=20, pop_size=15,
                 ind_mutpb=0.1, mutpb=0.25,cxpb=0.5, turns=3, hf_size=5):
        """

        :param preproc:
        :param classif:
        :param n_preproc:
        :param evaluator:
        :param ngen:
        :param pop_size:
        :param ind_mutpb:
        :param mutpb:
        :param cxpb:
        :param turns:
        :param hf_size:
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

        self.hf = tools.HallOfFame(hf_size)
        self._toolbox = self._toolbox_init()

        if evaluator is None:
            self._eval = DefaultEvaluator(self._toolbox)
        else:
            self._eval = evaluator

    # will be replaced by fit-predict pattern with GridSearch
    def fit(self, train_X, train_Y):
        """

        :param train_X:
        :param train_Y:
        :return:
        """
        self._eval.fit(train_X, train_Y)

        pop = self._toolbox.population(n=self._pop_size)

        fitnesses = map(self._toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(self._ngen):
            print("Gen {}:\n".format(g))
            print(pop)

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

        pipes = map(self._toolbox.compile, self.hf.items)
        scores = map(self._toolbox.evaluate, self.hf.items)

        return zip(pipes, scores)

    def _toolbox_init(self):
        """

        :return:
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

    def _ind_range(self):
        """

        :return:
        """
        i = 0
        while i < self.n_prep:
            i = i + 1
            yield random.randint(0, len(self.pr_l) - 1)

        yield random.randint(0, len(self.cl_l) - 1)

    def _compile_pipe(self, ind):
        """

        :param ind:
        :return:
        """

        valid_prepro = self.n_prep - ind.count(-1)
        valid_inds = np.where(ind != -1)

        prep = self.pr_l[valid_inds[:valid_prepro]]
        clf = self.cl_l[valid_inds[valid_prepro:]]
        return make_pipeline(*prep, *clf)
