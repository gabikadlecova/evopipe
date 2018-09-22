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


def mutate_individual(ind1, params, toolbox, index_pb, param_pb, swap_pb):
    """
    The indiviudal is mutated according to the probabilities. If a step is mutated, it is either replaced whole
    (with probability of swap_pb), or only some of its parameters are mutated (with probability param_pb for every
    parameter independently). The individual is modified in place.
    :param ind1: The individual to be mutated
    :param params: Dictionary which contains all method parameter dictionaries (indexed by name)
    :param toolbox: Deap toolbox
    :param index_pb: Probability of a single step to be mutated.
    :param param_pb: Probability of a single step parameter to be mutated.
    :param swap_pb: Probability of a step to be replaced by a random preprocessor/classifier. If it is not replaced,
                    the parameters are mutated instead.
    :return: A mutated individual
    """

    # list of indices on which the individual is mutated
    modif_ind = [i for i in range(0, len(ind1)) if random.random() < index_pb]

    for i in modif_ind:
        # swap mutation
        if random.random() < swap_pb:
            # random preprocessor or classifier respectively
            ind1[i] = toolbox.random_clf() if i == len(ind1) - 1 else toolbox.random_prepro()
        # parameter mutation
        else:
            name, p_list = ind1[i]
            ind1[i] = name, mutate_params(params[name], param_pb, p_list)

    return ind1


def mutate_params(possible, param_pb, values):
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
        ind.fitness.values = fit
        ind.train_test = tt

    # toolbox.log(population, 0)

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
            ind.fitness.values = fit
            ind.train_test = tt

        # next population is selected from the previous one and from produced offspring
        population[:] = toolbox.select(population + valid_offs, pop_size)

        hof.update(population)
        toolbox.log(population, g)

        if g % 5 == 0:
            print("\nGen {}:\n".format(g + 1))
            # current HallOfFame
            print("Hall of fame:")
            for hof_ind in hof.items:
                print(hof_ind)
