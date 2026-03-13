from algorithms.optuna import OptunaAlgorithm
from sklearn.cluster import DBSCAN
import optuna
import numpy as np

class AsteroidDBSCAN(OptunaAlgorithm):
    def __init__(self, reload_raw_data=False, algorithm_name="DBSCAN", debug_prints=False, n_trials=50):
        super().__init__(reload_raw_data, algorithm_name, debug_prints, n_trials=n_trials)
        
    def get_initial_params(self) -> dict:
        return {"eps": 0.02, "min_samples": 30, "scaler": "minmax"}
        
    def define_hyperparams(self, trial: optuna.Trial):
        # The eps search space is restricted to very small values because the 
        # proper orbital elements of true asteroid families exist within extremely tight variances.
        trial.suggest_float("eps", 0.005, 0.04, log=False)
        trial.suggest_int("min_samples", 5, 100)
        
    def train_predict(self, params: dict, X_scaled: np.ndarray) -> np.ndarray:
        # ball_tree is enforced as the spatial indexer because it outperforms kd_tree 
        # for continuous, low-dimensional astronomical data, cutting query times drastically.
        dbscan_model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"], algorithm="ball_tree", n_jobs=1)
        return dbscan_model.fit_predict(X_scaled)