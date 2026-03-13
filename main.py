from algorithms.kMeans import AsteroidKMeans
from algorithms.dbscan import AsteroidDBSCAN
from algorithms.hdbscan import AsteroidHDBSCAN

from tqdm import tqdm

trials = 25

models = [
    AsteroidKMeans(reload_raw_data=False, debug_prints=False, n_trials=trials),
    AsteroidHDBSCAN(reload_raw_data=False, debug_prints=False, n_trials=trials),
    AsteroidDBSCAN(reload_raw_data=False, debug_prints=False, n_trials=trials),
]

for i in tqdm(range(3)):
    model = models[i]
    model.benchmark()