import random


def cx_one_point_rev(ind1, ind2):
    """
    Performs a one point crossover of two individuals. The result individuals have each the length of the another,
    however, not the prefix, but the suffix part of the list is kept. Both individuals are modified in place.
    :param ind1: First individual
    :param ind2: Second individual
    :return: Offsprings of the crossover
    """
    min_len = min(len(ind1), len(ind2))
    right_len = random.randint(1, min_len)

    c_ind1 = len(ind1) - right_len
    c_ind2 = len(ind2) - right_len

    ind1[c_ind1:], ind2[c_ind2:] = ind2[c_ind2:], ind1[c_ind1:]
    return ind1, ind2


def cx_uniform(ind1, ind2, prepro_names, cx_swap_pb=0.3):
    pos1 = 0
    pos2 = 0

    names1 = list(map(lambda i: i[2], ind1[:-1]))
    names2 = list(map(lambda i: i[2], ind2[:-1]))

    prepro_ind1 = list(map(lambda x: x in names1, prepro_names))
    prepro_ind2 = list(map(lambda x: x in names2, prepro_names))

    for val1, val2 in zip(prepro_ind1, prepro_ind2):
        if not val1 and not val2:
            continue

        if not val1:
            if random.random() < cx_swap_pb:
                ind1.insert(pos1, ind2[pos2])
                ind2.pop(pos2)
                pos1 += 1
            else:
                pos2 += 1

        elif not val2:
            if random.random() < cx_swap_pb:
                ind2.insert(pos2, ind1[pos1])
                ind1.pop(pos1)
                pos2 += 1
            else:
                pos1 += 1

        else:
            if random.random() < cx_swap_pb:
                if ind1[pos1][0] == ind2[pos2][0]:
                    _cx_params(ind1[pos1][1], ind2[pos2][1], cx_swap_pb)
                else:
                    ind1[pos1], ind2[pos2] = ind2[pos2], ind1[pos1]
            pos1 += 1
            pos2 += 1

    # clf parameter crossover
    if ind1[-1][0] == ind2[-1][0]:
        if random.random() < cx_swap_pb:
            _cx_params(ind1[-1][1], ind2[-1][1], cx_swap_pb)

    return ind1, ind2


def _cx_params(params1, params2, cx_swap_pb):
    for key in params1.keys():
        if random.random() < cx_swap_pb:
            params1[key], params2[key] = params2[key], params1[key]

    return params1, params2


def mutate_individual(ind1, params, prepro_names, toolbox, index_pb, param_pb, swap_pb, len_pb, n_iter=4):
    """

    :param ind1:
    :param params:
    :param toolbox:
    :param index_pb:
    :param param_pb:
    :param swap_pb:
    :param len_pb:
    :param n_iter:
    :return:
    """

    if random.random() < swap_pb:
        ind1 = _mutate_once(ind1, params, toolbox, index_pb, 'swap')

    elif random.random() < len_pb:
        mutate_length(ind1, prepro_names, toolbox)

    else:
        ind1 = _mutate_params(ind1, toolbox, params, index_pb, param_pb, n_iter)

    return ind1


def _try_eval(ind, toolbox):
    fit, tt = toolbox.evaluate(ind)
    if fit is None:
        return False

    ind.fitness.values, ind.train_test = fit, tt
    return True


def _mutate_params(ind1, toolbox, params, index_pb, param_pb, n_iter):
    for i in range(n_iter):
        mutant = toolbox.clone(ind1)
        _mutate_once(mutant, params, toolbox, index_pb, 'params', param_pb=param_pb)

        # individual evaluation
        if not ind1.fitness.valid:
            if not _try_eval(ind1, toolbox):
                return ind1

        # mutant evaluation
        if not _try_eval(mutant, toolbox):
            continue

        # choose the better one
        if mutant.fitness.values[0] > ind1.fitness.values[0]:
            ind1 = mutant

    return ind1


def _mutate_once(ind1, params, toolbox, index_pb, method, param_pb=None):
    """
        The individual is mutated according to probabilities. Parameter mutation is performed using a hill-climbing
        algorithm. The individual is modified in place.
        :param ind1: The individual to be mutated
        :param params: Dictionary which contains all method parameter dictionaries (indexed by name)
        :param toolbox: Deap toolbox
        :param index_pb: Probability of a single step to be mutated.
        :param param_pb: Probability of a single step parameter to be mutated.
        :return: A mutated individual
        """

    # list of indices on which the individual is mutated
    modif_ind = [i for i in range(0, len(ind1)) if random.random() < index_pb]

    for i in modif_ind:
        # swap mutation
        if method == 'swap':
            # random preprocessor or classifier respectively
            ind1[i] = toolbox.random_clf() if i == len(ind1) - 1 else toolbox.random_prepro(ind1[i][2], used=ind1)

            # parameter mutation
        elif method == 'params':
            if i == len(ind1) - 1:
                name, p_list = ind1[i]
                ind1[i] = name, change_params(params[name], param_pb, p_list)
            else:
                name, p_list, p_type = ind1[i]
                ind1[i] = name, change_params(params[name], param_pb, p_list), p_type

        # invalid method
        else:
            raise ValueError

    return ind1


'''def mutate_length(ind, toolbox):
    # add preprocessor - individual too short/probability
    if len(ind) < 2 or random.random() < (1.0 / len(ind)):
        random_prepro = toolbox.random_prepro()
        ind.insert(0, random_prepro)
    # remove a preprocessor
    else:
        del ind[0]'''


def mutate_length(ind, prepro_names, toolbox):

    # insert random preprocessor
    if len(ind) < 2 or random.random() < (1.0 / len(ind)):
        names = list(map(lambda i: i[2], ind[:-1]))

        possible = list(filter(lambda val: val not in names, prepro_names))
        if len(possible) == 0:
            return

        prepro_name = random.choice(possible)
        prepro = toolbox.random_prepro(prepro_name)

        insert_at = 0
        for name in prepro_names:
            if name == prepro_name:
                break

            if name in names:
                insert_at += 1

        ind.insert(insert_at, prepro)

    # remove a preprocessor
    else:
        to_del = random.randint(0, len(ind) - 2)
        del ind[to_del]


def change_params(possible, param_pb, values):
    """
    Mutates some of the parameters. The values are replaced with a random choice from possible.
    :param possible: Dict of lists of possible values indexed by parameter name.
    :param param_pb: Probability of a single parameter to be mutated.
    :param values: Current values dict.
    :return: A mutated values dictionary.
    """
    # list of indices where the parameters are mutated
    modif_names = [name for name in values.keys() if random.random() < param_pb]
    for name in modif_names:
        values[name] = random.choice(possible[name])

    return values


def _test(ch):
    if len(ch) > 3:
        raise ValueError("len")

    names = list(map(lambda x: x[2], ch[:-1]))
    if len(names) > len(set(names)):
        raise ValueError("duplicate")

    if len(ch) == 3:
        if ch[0][2] == 'scaling':
            raise ValueError("first")


def simple_ea(population, toolbox, ngen, pop_size, cxpb, mutpb, hof):
    """
    Performs a simple evolutionary algorithm on population. The toolbox must contain following
    methods - clone(pop), mate(ind1, ind2), mutate(ind), evaluate(ind), select(pop, size), log(pop, gen_num).

    Prints the progress on standard output # todo verbosity parameter
    :param population: Starting population
    :param toolbox: Toolbox providing methods for the algorithm
    :param ngen: Number of generations
    :param pop_size: Population size in every generation
    :param cxpb: Crossover probability
    :param mutpb: Mutation probability
    :param hof: HallOfFame deap object # todo allow none
    :return: The result population after ngen generations
    """


    # setup
    scores = map(toolbox.evaluate, population)
    for ind, (fit, tt) in zip(population, scores):
        # skipping invalid values
        if fit is None:
            continue

        _test(ind)
        ind.fitness.values = fit
        ind.train_test = tt

    population[:] = [ind for ind in population if ind.fitness.valid]
    print('Evolution starting...')

    toolbox.log(population, 0)

    # evolution
    for g in range(1, ngen):

        # possible offspring
        offspring = toolbox.clone(population)

        # crossover
        for ch1, ch2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(ch1, ch2)
                del ch1.fitness.values
                del ch2.fitness.values

        # mutation
        for mut in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mut)
                del mut.fitness.values

        # offspring fitness update, these will be added to the population
        valid_offs = [ind for ind in offspring if not ind.fitness.valid]

        scores = map(toolbox.evaluate, valid_offs)
        for ind, (fit, tt) in zip(valid_offs, scores):
            if fit is None:
                continue

            ind.fitness.values = fit
            ind.train_test = tt

        offspring = [ind for ind in offspring if ind.fitness.valid]
        # next population is selected from the previous one and from produced offspring
        # population[:] = toolbox.select(population + offspring, pop_size)
        population[:] = hof[:2] + toolbox.select(offspring, pop_size - 2)

        hof.update(population)
        toolbox.log(population, g)

        print("\nGen {}:\n".format(g + 1))
        if g % 5 == 0:
            # current HallOfFame
            print("Hall of fame:")
            for hof_ind in hof.items:
                print(hof_ind)
