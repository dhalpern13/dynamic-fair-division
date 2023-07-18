import os
from collections import namedtuple
from itertools import product
from math import sqrt

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from algorithms import ranking_alg, ranking_alg_with_sample_epochs, compare_to_everything, welfare_maximizing, \
    quantile_maximizing

NUM_AGENTS = [2, 5]

FILE = 'data/dynamic-fair-division.csv'

# Up to 10^5, 100 timesteps.
TIME_STEPS = {int(i) for i in np.logspace(0, 5, num=100)}

TOTAL_ITEMS = {'total_items': 100_000}

# Epoch length parameters
EPOCHS = {
    'shorter': {
        'sampling_length_func': lambda epoch: epoch ** 2,
        'exploiting_length_func': lambda epoch: epoch ** 4
    },
    'longer': {
        'sampling_length_func': lambda epoch: epoch ** 4,
        'exploiting_length_func': lambda epoch: epoch ** 8
    },
}

# Epoch sampling parameters
EPOCH_SAMPLING = {
    'regular': {
        'sample_prob_func': lambda item, num_agents: sqrt(item) / item * num_agents if item / num_agents >= 20 else 1,
        'delete_prob_func': lambda item, num_agents: item ** (
                1 / 4) / item * num_agents if item / num_agents >= 20 else 0
    }
}

# Algorithms to run along with additional parameters.
ALGS = {
    'normal-short': (ranking_alg, {'compare_to_all': False} | EPOCHS['shorter'] | TOTAL_ITEMS),
    'normal-long': (ranking_alg, {'compare_to_all': False} | EPOCHS['longer'] | TOTAL_ITEMS),
    'epoch-compare-short': (ranking_alg, {'compare_to_all': True} | EPOCHS['shorter'] | TOTAL_ITEMS),
    'epoch-compare-long': (ranking_alg, {'compare_to_all': True} | EPOCHS['longer'] | TOTAL_ITEMS),
    'random-sampling': (ranking_alg_with_sample_epochs, EPOCH_SAMPLING['regular'] | TOTAL_ITEMS),
    'compare-to-all': (compare_to_everything, {'starting_sample': 20} | TOTAL_ITEMS),
    # 'deterministic-sqrt': (deterministic_squareroot, TOTAL_ITEMS),
    'welfare-max': (welfare_maximizing, TOTAL_ITEMS),
    'quantile-max': (quantile_maximizing, TOTAL_ITEMS),
}

MULTI_PROCESSSING = True

stats_columns = ['welfare', 'envy_12', 'envy_n1', 'ef', 'bundle_size_1', 'bundle_size_n', 'val_per_item_11',
                 'val_per_item_1n']
allocation_stats = namedtuple('allocation_stats', stats_columns)


def allocation_statistics(item_list, num_agents):
    """
    item_list represents an allocation as a list of (agent, item) tuples. Recall, an agent is just
    an integer {0,...,n-1} and an item is represented by an n-tuple of value per item.

    Returns a list of allocation_stats over time. I.e., stats[t].ef is 1 exactly when the allocation
    is 1 an time t. Similarly, stats[t].welfare is the welfare approximation at time t.
    """
    cur_max_welfare = 0
    cur_welfare = 0

    bundle_values = np.zeros((num_agents, num_agents))

    agent_1_cur_bundle_size = 0
    agent_n_cur_bundle_size = 0

    stats = []

    for receiving_agent, item in item_list:
        cur_max_welfare += max(item)
        cur_welfare += item[receiving_agent]

        for agent in range(num_agents):
            bundle_values[agent, receiving_agent] += item[agent]

        is_ef = int(all(bundle_values[i, i] >= bundle_values[i, j] for i, j in product(range(num_agents), repeat=2)))

        if receiving_agent == 0:
            agent_1_cur_bundle_size += 1
        elif receiving_agent == num_agents - 1:
            agent_n_cur_bundle_size += 1

        val_per_item_11 = bundle_values[0, 0] / agent_1_cur_bundle_size if agent_1_cur_bundle_size > 0 else 0
        val_per_item_1n = bundle_values[0, -1] / agent_n_cur_bundle_size if agent_n_cur_bundle_size > 0 else 0

        stats.append(allocation_stats(welfare=cur_welfare / cur_max_welfare,
                                      envy_12=bundle_values[0, 0] - bundle_values[0, 1],
                                      envy_n1=bundle_values[-1, -1] - bundle_values[-1, 0],
                                      ef=is_ef,
                                      bundle_size_1=agent_1_cur_bundle_size,
                                      bundle_size_n=agent_n_cur_bundle_size,
                                      val_per_item_11=val_per_item_11,
                                      val_per_item_1n=val_per_item_1n))

    return stats


def run_on_dataset(dataset_file):
    """
    Run all algorithms on dataset contained in item_vals
    """
    dataset = np.load(f'item_vals/{dataset_file}')
    dataset_name = dataset_file.rsplit('.')[0]  # Remove extension

    rows = []
    for num_agents, (alg_name, (alg, extra_params)) in product(NUM_AGENTS, ALGS.items()):
        item_list = alg(num_agents, dataset, **extra_params)
        stats = allocation_statistics(item_list, num_agents)
        for timestep, stats in enumerate(stats, start=1):
            if timestep not in TIME_STEPS:
                continue
            rows.append({
                            'dataset': dataset_name,
                            'num_agents': num_agents,
                            'alg': alg_name,
                            'timestep': timestep
                        } | stats._asdict())
    pd.DataFrame(data=rows).to_csv(FILE, index=False, float_format='%g', mode='a', header=False)


def main():
    pd.DataFrame(
        columns=['dataset', 'num_agents', 'alg', 'timestep'] + stats_columns).to_csv(FILE, index=False)
    files = os.listdir('item_vals')
    if MULTI_PROCESSSING:
        process_map(run_on_dataset, files, chunksize=5)
    else:
        for data_set in tqdm(files):
            run_on_dataset(data_set)


if __name__ == '__main__':
    main()
