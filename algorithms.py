from itertools import islice, count
from math import sqrt
from random import random, randint, choice, seed

from sortedcontainers import SortedList

seed(0)

"""
Algorithms are given a number of agents, Numpy mxn dataset, and a total number of items along with other parameters.
Items are represented by an n-tuple of each agents value.

Algorithms return a list of (agent, item) tuples,
i.e., [(0, (.5,.6)), (0, (.6, .5)), (1, (.5, .4))] represents an allocation of 3 items,
the first two to agent 0 and the third to agent 1.

In most algorithms, must give their ranking of a fresh item among a previously given sample.
Samples are stored using a SortedList of sample values.
The function list.bisect_left(val) returns the number of items in the sample valued less than the current number.
"""


def dataset_sampler(dataset, num_agents, total_items):
    """
    Turns Numpy dataset into generator over items
    """
    for item in islice(dataset, 0, total_items):
        yield tuple(item[:num_agents])


def welfare_maximizing(num_agents, dataset, total_items):
    """
    Give item to value-maximizing agent, breaking ties randomly
    """
    items = []
    for item in dataset_sampler(dataset, num_agents, total_items):
        max_val = max(item)
        max_agents = [i for i in range(num_agents) if item[i] == max_val]
        receiving_agent = choice(max_agents)
        items.append((receiving_agent, item))

    return items


def quantile_maximizing(num_agents, dataset, total_items):
    """
    Give item to quantile-maximizing agent, breaking ties randomly.
    Essentially treats entire dataset as "sample" items.
    """
    samples = [SortedList(dataset.T[i]) for i in range(num_agents)]
    items = []
    for item in dataset_sampler(dataset, num_agents, total_items):
        rankings = [samples[i].bisect_left(item[i]) / len(samples[i]) for i in range(num_agents)]
        max_ranking = max(rankings)
        max_agents = [i for i in range(num_agents) if rankings[i] == max_ranking]
        receiving_agent = choice(max_agents)
        items.append((receiving_agent, item))
        samples[receiving_agent].add(item[receiving_agent])

    return items


def ranking_alg(num_agents, dataset, total_items, sampling_length_func, exploiting_length_func, compare_to_all):
    """
    Run standard ranking algorithm.
    
    Parameters sampling_length_func and exploiting_length_func are functions that given an epoch return
    the sampling/exploiting length of that epoch. I.e., Algorithm 1 has sampling_length_func(k) = k^4
    and exploiting_length_func(k) = k^8.

    If compare_to_all is true, then keep adding items to the sample even in the exploiting phase.
    """
    dataset_iter = dataset_sampler(dataset, num_agents, total_items)
    items = []
    try:
        for epoch in count(1):
            samples = [SortedList() for _ in range(num_agents)]
            # Give sampling_length(epoch) number of items to each agents.
            for _ in range(sampling_length_func(epoch)):
                for i in range(num_agents):
                    new_item = next(dataset_iter)
                    samples[i].add(new_item[i])
                    items.append((i, new_item))
            for _ in range(exploiting_length_func(epoch)):
                new_item = next(dataset_iter)
                rankings = [samples[i].bisect_left(new_item[i]) / len(samples[i]) for i in range(num_agents)]
                max_ranking = max(rankings)
                max_agents = [i for i in range(num_agents) if rankings[i] == max_ranking]
                receiving_agent = choice(max_agents)
                items.append((receiving_agent, new_item))
                if compare_to_all:
                    samples[receiving_agent].add(new_item[receiving_agent])
    except StopIteration:
        # No more items left in dataset. Return current allocation
        return items


def ranking_alg_with_sample_epochs(num_agents, dataset, total_items, sample_prob_func, delete_prob_func):
    items = []
    samples = [SortedList() for _ in range(num_agents)]
    for item_num, new_item in enumerate(dataset_sampler(dataset, num_agents, total_items)):
        if random() < delete_prob_func(item_num, num_agents):
            agent = randint(0, num_agents - 1)
            agent_samples = samples[agent]
            item = choice(agent_samples)
            agent_samples.remove(item)

        if random() < sample_prob_func(item_num, num_agents):
            agent = randint(0, num_agents - 1)
            samples[agent].add(new_item[agent])
            items.append((agent, new_item))
        else:
            rankings = [samples[i].bisect_left(new_item[i]) / len(samples[i]) for i in range(num_agents)]
            max_ranking = max(rankings)
            max_agents = [i for i in range(num_agents) if rankings[i] == max_ranking]
            receiving_agent = choice(max_agents)
            items.append((receiving_agent, new_item))

    return items


def deterministic_squareroot(num_agents, dataset, total_items):
    items = []
    samples = [SortedList() for _ in range(num_agents)]

    sample_times = {t * num_agents for t in range(num_agents)} | {t ** 2 for t in
                                                                  range(num_agents, int(sqrt(total_items)) + 1)}
    agent_sample_times = {}
    for i in range(num_agents):
        for t in sample_times:
            agent_sample_times[t + i] = i

    to_delete = {}

    for timestep, new_item in enumerate(dataset_sampler(dataset, num_agents, total_items), start=1):
        if (delete_agent_item := to_delete.get(timestep)) is not None:
            delete_agent, delete_item = delete_agent_item
            samples[delete_agent].remove(delete_item)

        if (sample_agent := agent_sample_times.get(timestep)) is not None:
            samples[sample_agent].add(new_item[sample_agent])
            to_delete[2 * timestep] = (sample_agent, new_item[sample_agent])
            items.append((sample_agent, new_item))
        else:
            rankings = [samples[i].bisect_left(new_item[i]) / len(samples[i]) for i in range(num_agents)]
            max_ranking = max(rankings)
            max_agents = [i for i in range(num_agents) if rankings[i] == max_ranking]
            receiving_agent = choice(max_agents)
            items.append((receiving_agent, new_item))

    return items


def compare_to_everything(num_agents, dataset, total_items, starting_sample=20):
    items = []
    dataset_iter = dataset_sampler(dataset, num_agents, total_items)
    samples = [SortedList() for _ in range(num_agents)]
    for _ in range(starting_sample):
        for i in range(num_agents):
            new_item = next(dataset_iter)
            samples[i].add(new_item[i])
            items.append((i, new_item))
    for item in dataset_iter:
        rankings = [samples[i].bisect_left(item[i]) / len(samples[i]) for i in range(num_agents)]
        max_ranking = max(rankings)
        max_agents = [i for i in range(num_agents) if rankings[i] == max_ranking]
        receiving_agent = choice(max_agents)
        items.append((receiving_agent, item))
        samples[receiving_agent].add(item[receiving_agent])

    return items
