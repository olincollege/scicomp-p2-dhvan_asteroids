from algorithms.kMeans import AsteroidKMeans
from algorithms.dbscan import AsteroidDBSCAN
from algorithms.hdbscan import AsteroidHDBSCAN
from tqdm import tqdm

correct_families = []
# for i in tqdm(range(10)):
# model = AsteroidKMeans(debug_prints = True)
# model = AsteroidHDBSCAN(debug_prints=True)
model = AsteroidDBSCAN(debug_prints=True)
# correct_families.append(model.benchmark())
print(model.benchmark())

# print(sum(correct_families) / len(correct_families))

# model.visualize_clusters()