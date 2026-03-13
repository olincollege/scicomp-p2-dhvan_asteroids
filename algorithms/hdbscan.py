from algorithms.optuna import OptunaAlgorithm
import hdbscan
import optuna
import numpy as np

class AsteroidHDBSCAN(OptunaAlgorithm):
    def __init__(self, reload_raw_data=False, algorithm_name="HDBSCAN", debug_prints=False, n_trials=50):
        # Force n_jobs=1 at the wrapper level to prevent parallelization conflicts 
        # between Optuna trials and the HDBSCAN C-backend.
        super().__init__(reload_raw_data, algorithm_name, debug_prints, n_trials=n_trials, n_jobs=1)
        
    def get_initial_params(self) -> dict:
        return {"min_cluster_size": 15, "min_samples": 5, "scaler": "standard"}
        
    def define_hyperparams(self, trial: optuna.Trial):
        trial.suggest_int("min_cluster_size", 5, 100)
        trial.suggest_int("min_samples", 1, 50)
        
    def train_predict(self, params: dict, X_scaled: np.ndarray) -> np.ndarray:
        # core_dist_n_jobs=-1 isolates the parallelization solely to the most 
        # computationally expensive step of HDBSCAN (the distance matrix calculation).
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=params["min_cluster_size"], min_samples=params["min_samples"], core_dist_n_jobs=-1)
        return hdbscan_model.fit_predict(X_scaled)