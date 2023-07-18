# dynamic-fair-division
Content
-------
- [aglorithms.py](algorithms.py)
  Implementation of all the algorithms we test.
- [distribution_val_generator.py](distribution_val_generator.py) Generates datasets of agent values from distributions. Stores them as numpy ndarrays in item_vals/ folder.
- [dataset_val_generator.py](dataset_val_generator.py) Generates datasets of agent values from real datasets. Stores them as numpy ndarrays in item_vals/ folder. Note to run this data must be downloaded and placed in data/ folder
- [experiment.py](experiment.py) Runs the experiments and stores the results in [data/dynamic-fair-division3.csv](data/dynamic-fair-division3.csv).
- [plots.ipynb](plots.ipynb) is an IPython notebook that generates the plots using the data.
  
Software requirements
---------------------
Run using
- Python 3.10.12
- joblib 1.2.0
- lenskit 0.14.2
- numpy 1.23.5
- pandas 1.5.2
- scipy 1.11.1
- sortedcontainers 2.4.0
- tqdm 4.64.1
