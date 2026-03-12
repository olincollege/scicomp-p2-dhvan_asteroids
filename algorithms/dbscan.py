from algorithms.algorithm import Algorithm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from decorators import timer
import numpy as np

class AsteroidDBSCAN(Algorithm):
    def __init__(self, reload_raw_data = False, algorithm_name = "DBSCAN", debug_prints = False):
        self.dbscan_model = DBSCAN(eps=0.05, min_samples=30, algorithm="ball_tree")
        self.scaler = MinMaxScaler()
        super().__init__(reload_raw_data, algorithm_name, debug_prints)
        
    @timer
    def fit_predict(self):
        X_scaled = self.scaler.fit_transform(self.X)
        predictions = self.dbscan_model.fit_predict(X_scaled)
        
        print("\nCluster Breakdown:", np.unique(predictions, return_counts=True))
        
        self.cached_predictions = predictions
        
        return predictions
