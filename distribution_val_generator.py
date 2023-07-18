import os

import numpy as np
from scipy.stats import beta, uniform

np.random.seed(0)

REGULAR = {
    "BETA_1_1": beta(1, 1),
    "BETA_3_3": beta(3, 3),
    "BETA_6_6": beta(6, 6),
    "BETA_10_10": beta(10, 10),
    "BETA_1_3": beta(1, 3),
    "BETA_1_6": beta(1, 6),
    "BETA_1_10": beta(1, 10),
    "BETA_3_1": beta(3, 1),
    "BETA_6_1": beta(6, 1),
    "BETA_10_1": beta(10, 1),
}

COMMON = {
    "UNIF_COR_2_8": (uniform(0, .2), uniform(0, .8)),
    "UNIF_COR_5_5": (uniform(0, .5), uniform(0, .5)),
    "UNIF_COR_8_2": (uniform(0, .8), uniform(0, .2))
}

if __name__ == '__main__':
    n = 5
    m = 100_000
    if not os.path.exists('item_vals'):
        os.mkdir('item_vals')

    for iteration in range(100):
        for name, dist in REGULAR.items():
            sample = dist.rvs((m, n))
            np.save(f'item_vals/{name}-{iteration}.npy', sample)
        for name, (common, noise) in COMMON.items():
            common_vals = common.rvs((m, 1))
            noise_vals = noise.rvs((m, n))
            sample = common_vals + noise_vals
            np.save(f'item_vals/{name}-{iteration}.npy', sample)
