from abc import ABC, abstractmethod
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
import os
from constants import TRAIN_TEST_VALIDATE_SPLIT
from typing import Self


class Algorithm(ABC):
    
    def __init__(self, reload_raw_data, algorithm_name):
        self.algorithm_name = algorithm_name
        os.makedirs("saved_obj", exist_ok=True)

        if(reload_raw_data):
            print("RELOADING RAW DATA")
            _time = time.perf_counter()
            complete_asteroid_dataset = pd.read_csv('raw_data/asteroids.csv', skiprows=2, sep=r'[\s;]+', engine='python')
            families_dataset = pd.read_csv('raw_data/families.csv', sep=r'\s+', engine='python')
            
            complete_asteroid_dataset['no'] = complete_asteroid_dataset['no'].astype(str)
            families_dataset['%ast.name'] = families_dataset['%ast.name'].astype(str)
            
            raw_merged_df = pd.merge(complete_asteroid_dataset, families_dataset, left_on="no", right_on="%ast.name", how="inner")
            dataset = raw_merged_df[['a', 'ecc', 'sinI', 'family1']]
            train, test, validate = TRAIN_TEST_VALIDATE_SPLIT
            _dataset, test_dataset = train_test_split(dataset, test_size=test)
            
            relative_validate = validate / (train + validate)
            
            train_dataset, validate_dataset = train_test_split(_dataset, test_size=relative_validate)
            
            with open('saved_obj/complete_asteroid_dataset.pkl', 'wb') as f:
                pickle.dump(complete_asteroid_dataset, f)
                
            with open('saved_obj/families_dataset.pkl', 'wb') as f:
                pickle.dump(families_dataset, f)
                
            with open('saved_obj/dataset.pkl', 'wb') as f:
                pickle.dump(dataset, f)
                
            with open('saved_obj/train_dataset.pkl', 'wb') as f:
                pickle.dump(train_dataset, f)
                
            with open('saved_obj/test_dataset.pkl', 'wb') as f:
                pickle.dump(test_dataset, f)
                
            with open('saved_obj/validate_dataset.pkl', 'wb') as f:
                pickle.dump(validate_dataset, f)
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
                
            with open('saved_obj/train_dataset.pkl', 'rb') as f:
                train_dataset = pickle.load(f)
                
            with open('saved_obj/test_dataset.pkl', 'rb') as f:
                test_dataset = pickle.load(f)
                
            with open('saved_obj/validate_dataset.pkl', 'rb') as f:
                validate_dataset = pickle.load(f)
            print(f'Saved files loaded in {time.perf_counter() - _time:.3f}s')
        
        self.dataset: pd.DataFrame = dataset
        self.test_dataset: pd.DataFrame = test_dataset
        self.train_dataset: pd.DataFrame = train_dataset
        self.validate_dataset: pd.DataFrame = validate_dataset
        
        self.X_train: pd.DataFrame = train_dataset[['a', 'ecc', 'sinI']]
        self.Y_train: pd.DataFrame = train_dataset['family1']
        
        self.X_test: pd.DataFrame = test_dataset[['a', 'ecc', 'sinI']]
        self.Y_test: pd.DataFrame = test_dataset['family1']
        
        self.X_validate: pd.DataFrame = validate_dataset[['a', 'ecc', 'sinI']]
        self.Y_validate: pd.DataFrame = validate_dataset['family1']
        
        self.complete_asteroid_dataset: pd.DataFrame = complete_asteroid_dataset
        self.families_dataset: pd.DataFrame = families_dataset
        
        self.cached_predictions = None
        
    @abstractmethod
    def fit(self) -> Self:
        pass
    
    @abstractmethod
    def predict(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def validate(self) -> np.ndarray:
        pass        
    
    def truncate(self, dataframe: pd.DataFrame, start_idx: int = 0, end_idx: int | None = None) -> pd.DataFrame:
        if(end_idx is None):
            return dataframe.iloc[start_idx:]
        
        return dataframe.iloc[start_idx:end_idx]
    
    def confusion_matrix(self,) -> np.ndarray:
        predictions = self.predict() if self.cached_predictions is None else self.cached_predictions
        return confusion_matrix(self.Y_test, predictions)
    
    def accuracy(self) -> float:
        predictions = self.predict() if self.cached_predictions is None else self.cached_predictions
        return float(accuracy_score(self.Y_test, predictions))
    
    def precision(self) -> np.ndarray:
        predictions = self.predict() if self.cached_predictions is None else self.cached_predictions
        return precision_score(self.Y_test, predictions, average=None, zero_division = 0) # type: ignore
    
    def benchmark(self) -> int:
        precision = self.precision()
        correct = np.count_nonzero(precision >= 0.95)
        return int(correct)