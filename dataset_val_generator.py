# %% import and general configure
import numpy as np
import pandas as pd
# import pyomo.environ as pyo
from lenskit import util
from lenskit.algorithms import als
from lenskit.batch import predict
from lenskit.metrics.predict import user_metric, rmse
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
import time
# import os
# os.environ['MKL_THREADING_LAYER'] = 'tbb'
#
# pd.set_option('display.width', 300)
# pd.set_option('display.max_columns', 15)
# np.set_printoptions(linewidth=300)
# np.set_printoptions(precision=4)
# np.set_printoptions(suppress=False)

rng = np.random.default_rng(0)

# %% read data


def read_rent_the_runway():
    """This function will likely have to be revised for different datasets. The output needs to have (user, item,
    rating) columns. """
    df = pd.read_json('data/renttherunway.json', lines=True)    # change this to where the dataset is
    data = df[['user_id', 'item_id', 'rating']].dropna()
    data.columns = ['user', 'item', 'rating']
    print('Reading rent the runway')
    return data


def read_amazon(file='auto'):
    """This function will likely have to be revised for different datasets. The output needs to have (uer, item,
    rating) columns. """
    filenames = {'auto':'Automotive_5', 'instr':'Musical_Instruments_5'}
    assert file in filenames.keys()
    df = pd.read_json('data/'+filenames[file]+'.json', lines=True)    # change this to where the dataset is
    data = df[['reviewerID', 'asin', 'overall']].dropna()
    data.columns = ['user', 'item', 'rating']
    print('Reading amazon:', file)

    return data

def read_recipes():
    df = pd.read_csv('data/RAW_interactions.csv')
    data = df[['user_id', 'recipe_id', 'rating']].dropna()
    data.columns = ['user', 'item', 'rating']
    return data


def data_summary(data):
    return data.user.nunique(), data.item.nunique(), data.rating.count(), \
           data.rating.min(), data.rating.quantile(.05), data.rating.quantile(.50), \
           data.rating.mean(), data.rating.quantile(.95), data.rating.max()

def sample_est(rs, data, n, m, seed=None):
    rng = np.random.default_rng(seed)
    print('Running with ', m, ' items and ', n, ' users.')
    users = rng.choice(data['user'].unique(), n, replace=False)
    items = rng.choice(data['item'].unique(), m, replace=True)
    ua, ia = np.meshgrid(users, items)
    test = pd.DataFrame({'user': ua.ravel(), 'item': ia.ravel()})
    preds = predict(rs, test, n_jobs=1)
    ratings = preds['prediction'].values.reshape(n, m)  # dense predicted rating matrix
    return ratings

def run_on_data(data, name, n=5, m=100_000, iterations=100):
    print(name, data_summary(data))

    # %% train and test a rating predictor
    rs = als.BiasedMF(50)
    rs.fit(data)

    for iteration in range(iterations):
        ratings = sample_est(rs, data, n, m).T
        np.random.shuffle(ratings)

        np.save(f'item_vals/{name}-{iteration}.npy', (ratings / ratings.max()))
        Vs = np.exp(ratings)  # some papers do value = e^rating
        np.save(f'item_vals/{name}_exp-{iteration}.npy', (Vs / Vs.max()))


#data = read_rent_the_runway()
amazon_data = read_amazon()
run_on_data(amazon_data, 'auto')
recipe_data = read_recipes()
run_on_data(recipe_data, 'recipes')
