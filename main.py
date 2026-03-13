from algorithms.kMeans import AsteroidKMeans
from algorithms.dbscan import AsteroidDBSCAN
from algorithms.hdbscan import AsteroidHDBSCAN

trials = 25

models = [
    (AsteroidKMeans(reload_raw_data=False, debug_prints=False, n_trials=trials), "KMEANS"),
    (AsteroidHDBSCAN(reload_raw_data=False, debug_prints=False, n_trials=trials), "HDBSCAN"),
    (AsteroidDBSCAN(reload_raw_data=False, debug_prints=False, n_trials=trials), "DBSCAN"),
]

benchmarks = []

for i in range(3):
    model = models[i][0]
    benchmarks.append(model.benchmark())
    
for i, benchmark in enumerate(benchmarks):
    print(f'Number of families above 95% completeness for model: {models[i][1]}: {benchmark[1]}')