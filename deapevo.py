import numpy as np
import random
from deap import tools


def cx_one_point_rev(ind1, ind2):
    min_len = min(len(ind1), len(ind2))
    right_len = random.randint(1, min_len)

    c_ind1 = len(ind1) - right_len
    c_ind2 = len(ind2) - right_len

    ind1[c_ind1:], ind2[c_ind2:] = ind2[c_ind2:], ind1[c_ind1:]
    return ind1, ind2


def mutate_individual(params, index_pb, param_pb, ind1):
    modif_ind = [i for i in range(0, len(ind1)) if random.random() < index_pb]
    for i in modif_ind:
        name, p_list = ind1[i]
        ind1[i] = mutate_params(params[name], param_pb, p_list)

    return ind1


def mutate_params(possible, param_pb, values):
    modif_params = [p for p in range(0, len(values)) if random.random() < param_pb]
    for p in modif_params:
        name, _ = values[p]
        values[p] = random.choice(possible[name])

    return values


def simple_ea(population, toolbox):
    pass


def stats_init():
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('var', np.var)
    stats.register('min', np.min)
    stats.register('max', np.max)
    return stats

