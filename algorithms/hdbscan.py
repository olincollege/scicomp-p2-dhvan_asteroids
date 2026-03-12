from algorithms.algorithm import Algorithm
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from decorators import timer


class AsteroidHDBSCAN(Algorithm):
    def __init__(self, reload_raw_data = False, algorithm_name = "DBSCAN", debug_prints = False):
        self.hdbscan_model = HDBSCAN(min_cluster_size = 5)
        self.scaler = StandardScaler()
        super().__init__(reload_raw_data, algorithm_name, debug_prints)
        
    @timer
    def fit_predict(self):
        X_scaled = self.scaler.fit_transform(self.X)
        predictions = self.hdbscan_model.fit_predict(X_scaled)
        
        self.cached_predictions = predictions
        
        return predictions
