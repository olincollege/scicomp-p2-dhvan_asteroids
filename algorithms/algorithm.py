from abc import ABC, abstractmethod
import builtins
import pandas as pd
import pickle
from sklearn.metrics import completeness_score
from sklearn.metrics.cluster import contingency_matrix
import time
import numpy as np
import os
from scipy.optimize import linear_sum_assignment


class Algorithm(ABC):
    
    def __init__(self, reload_raw_data, algorithm_name, debug_prints):
        self.algorithm_name = algorithm_name
        os.makedirs("saved_obj", exist_ok=True)
        
        if(not debug_prints):
            print = lambda *args, **kwargs: None
        else:
            print = builtins.print

        if(reload_raw_data):
            print("RELOADING RAW DATA")
            _time = time.perf_counter()
            complete_asteroid_dataset = pd.read_csv('raw_data/asteroids.csv', skiprows=2, sep=r'[\s;]+', engine='python')
            families_dataset = pd.read_csv('raw_data/families.csv', sep=r'\s+', engine='python')
            
            complete_asteroid_dataset['no'] = complete_asteroid_dataset['no'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
            families_dataset['%ast.name'] = families_dataset['%ast.name'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
            
            raw_merged_df = pd.merge(complete_asteroid_dataset, families_dataset, left_on="no", right_on="%ast.name", how="left")
            dataset = raw_merged_df[['a', 'ecc', 'sinI', 'family1']]

            with open('saved_obj/complete_asteroid_dataset.pkl', 'wb') as f:
                pickle.dump(complete_asteroid_dataset, f)
                
            with open('saved_obj/families_dataset.pkl', 'wb') as f:
                pickle.dump(families_dataset, f)
                
            with open('saved_obj/dataset.pkl', 'wb') as f:
                pickle.dump(dataset, f)

            print(f'Data loaded and processed in {time.perf_counter() - _time}s')
        else:
            print("LOADING SAVED PICKLE FILES")
            _time = time.perf_counter()
            with open('saved_obj/complete_asteroid_dataset.pkl', 'rb') as f:
                complete_asteroid_dataset = pickle.load(f)
            
            with open('saved_obj/families_dataset.pkl', 'rb') as f:
                families_dataset = pickle.load(f)
                
            with open('saved_obj/dataset.pkl', 'rb') as f:
                dataset = pickle.load(f)

            print(f'Saved files loaded in {time.perf_counter() - _time:.3f}s')
        
        self.dataset: pd.DataFrame = dataset

        self.X: pd.DataFrame = dataset[['a', 'ecc', 'sinI']]
        self.Y: pd.Series = dataset['family1']
                
        self.datasets = {
            "X": self.X,
            "Y": self.Y,
        }
        
        self.complete_asteroid_dataset: pd.DataFrame = complete_asteroid_dataset
        self.families_dataset: pd.DataFrame = families_dataset
        
        self.cached_predictions = None
        
        self.fit_predict()

    @abstractmethod
    def fit_predict(self) -> np.ndarray:
        pass
    
    def truncate(self, dataframe: pd.DataFrame, start_idx: int = 0, end_idx: int | None = None) -> pd.DataFrame:
        if(end_idx is None):
            return dataframe.iloc[start_idx:]
        
        return dataframe.iloc[start_idx:end_idx]
        
    def completeness(self) -> float:
        predictions = self.fit_predict() if self.cached_predictions is None else self.cached_predictions
        
        return float(completeness_score(self.Y, predictions))
    
    def benchmark(self) -> int:
        predictions = self.fit_predict() if self.cached_predictions is None else self.cached_predictions
        
        Y_eval = self.Y.fillna("ZZZ_Background").astype(str)
        
        c_matrix = np.array(contingency_matrix(Y_eval, predictions))
        
        row_idx, col_idx = linear_sum_assignment(-c_matrix)
        
        optimal_matches = c_matrix[row_idx, col_idx]
        
        family_totals = c_matrix.sum(axis = 1)[row_idx]
        cluster_totals = c_matrix.sum(axis = 0)[col_idx]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # Completeness is a metric that tracks the amount of a family that is together in a cluster
            completeness_ratios = optimal_matches / family_totals
            # Purity is a metric that tracks the amount of another family that is in the cluster
            purity_ratios = optimal_matches / cluster_totals
    
        # Want completeness and purity to both be above 95% since that indicates that: 
        # 95% of the famliy stuck together in the cluster, and 95% of the cluster is made up of just that family
        successful_families = np.count_nonzero(
            (completeness_ratios >= 0.95) &
            (purity_ratios >= 0.70) &
            (row_idx != c_matrix.shape[0] - 1) & # 3. Ignore the Background row (last row)
            (col_idx != 0)                       # 4. Ignore the Noise column (first column)
        )
        
        return int(successful_families)