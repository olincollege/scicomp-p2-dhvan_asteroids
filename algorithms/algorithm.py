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
    """
    Base class for executing and evaluating unsupervised clustering algorithms 
    on asteroid orbital parameter data to identify asteroid families.
    """
    def __init__(self, reload_raw_data, algorithm_name, debug_prints):
        self.algorithm_name = algorithm_name
        os.makedirs("saved_obj", exist_ok=True)
        
        # Globally suppress prints if debug is disabled to prevent console spam during parallel execution
        if(not debug_prints):
            print = lambda *args, **kwargs: None
        else:
            print = builtins.print

        if(reload_raw_data):
            print("RELOADING RAW DATA")
            _time = time.perf_counter()
            
            # The raw data contains variable whitespace and trailing decimals in IDs
            # that prevent a clean relational join between the structural and family datasets.
            complete_asteroid_dataset = pd.read_csv('raw_data/asteroids.csv', skiprows=2, sep=r'[\s;]+', engine='python')
            families_dataset = pd.read_csv('raw_data/families.csv', sep=r'\s+', engine='python')
            
            complete_asteroid_dataset['no'] = complete_asteroid_dataset['no'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
            families_dataset['%ast.name'] = families_dataset['%ast.name'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
            
            raw_merged_df = pd.merge(complete_asteroid_dataset, families_dataset, left_on="no", right_on="%ast.name", how="left")

            # RFL, QCM, and QCO are observation quality flags. 
            # Non-zero values indicate unreliable measurements that introduce physical impossibilities 
            # or extreme noise into the clustering space. Filtering them out guarantees a stable baseline.
            raw_merged_df = raw_merged_df[
                (raw_merged_df['RFL'] == 0) &
                (raw_merged_df['QCM'] == 0) &
                (raw_merged_df['QCO'] == 0)
            ]

            # We restrict the feature space to Proper Elements (a, ecc, sinI) and Frequencies (g, s).
            # These are dynamically invariant over millions of years, causing true family members 
            # to cluster tightly together, while background objects remain scattered.
            dataset = raw_merged_df[['a', 'ecc', 'sinI', 'g', 's', 'family1']]

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

        self.X: pd.DataFrame = dataset[['a', 'ecc', 'sinI', 'g', 's']]
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

    def benchmark(self) -> tuple[int, int]:
        """
        Evaluates the clustering performance by mapping unsupervised clusters to the true asteroid families.
        """
        predictions = self.fit_predict() if self.cached_predictions is None else self.cached_predictions
        
        Y_eval = self.Y.fillna("ZZZ_Background").astype(str)
        
        c_matrix = np.array(contingency_matrix(Y_eval, predictions))
        
        # The Hungarian algorithm guarantees the mathematically optimal 1:1 mapping 
        # between predicted clusters and true ground-truth families by minimizing the negative weights.
        row_idx, col_idx = linear_sum_assignment(-c_matrix)
        
        optimal_matches = c_matrix[row_idx, col_idx]
        
        family_totals = c_matrix.sum(axis = 1)[row_idx]
        cluster_totals = c_matrix.sum(axis = 0)[col_idx]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            completeness_ratios = optimal_matches / family_totals
            purity_ratios = optimal_matches / cluster_totals
            
        # We discard mappings involving the background noise class (last row due to "ZZZ") 
        # or the unclustered noise bin (column 0 in sklearn algorithms) 
        # because we are only grading the algorithm on discrete family recovery.
        valid_mask = (row_idx != c_matrix.shape[0] - 1) & (col_idx != 0)
        
        # Count the number of families that pass the benchmark
        # Completeness >= 95%, Purity >= 80% and aren't noise
        successful_families = np.count_nonzero(
            (completeness_ratios >= 0.95) &
            (purity_ratios >= 0.80) &
            valid_mask
        )
        
        complete_families = np.count_nonzero(
            (completeness_ratios >= 0.95) &
            valid_mask
        )
        
        return int(successful_families), int(complete_families)