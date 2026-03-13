import os
# These environment variables must be injected before numpy/sklearn load.
# They lock the underlying BLAS/OpenMP C-libraries to a single thread to prevent 
# catastrophic CPU thrashing and deadlocks when Optuna runs multiple independent trials simultaneously.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from algorithms.optuna import OptunaAlgorithm
from sklearn.cluster import KMeans
import optuna
import numpy as np

class AsteroidKMeans(OptunaAlgorithm):
    def __init__(self, reload_raw_data=False, algorithm_name="KMeans", debug_prints=False, n_trials=50):
        super().__init__(reload_raw_data, algorithm_name, debug_prints, n_trials=n_trials)
        
    def get_initial_params(self) -> dict:
        return {"n_clusters": 50, "scaler": "standard"}
        
    def define_hyperparams(self, trial: optuna.Trial):
        # The upper bound reflects the estimated upper limit of discrete, statistically significant 
        # asteroid families present in the main belt data.
        trial.suggest_int("n_clusters", 10, 200)
        
    def train_predict(self, params: dict, X_scaled: np.ndarray) -> np.ndarray:
        kmeans_model = KMeans(n_clusters=params["n_clusters"], random_state=42, n_init="auto")
        return kmeans_model.fit_predict(X_scaled)