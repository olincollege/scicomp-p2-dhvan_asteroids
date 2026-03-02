from typing import Self

import pandas as pd
from pandas import DataFrame as df
from typing import Dict
from algorithms.algorithm import Algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from decorators import timer

class AsteroidLogisticRegression(Algorithm):
    
    datasets: Dict[str, df]

    def __init__(self, reload_raw_data = False, algorithm_name = "LogisticRegression"):
        self.scaler = StandardScaler()
        self.lr_model = LogisticRegression(max_iter=1000)
        
        super().__init__(reload_raw_data, algorithm_name)
        
    @timer
    def fit(self):
        return self.lr_model.fit(self.scaler.fit_transform(self.datasets["X_train"]), self.datasets["Y_train"])
    
    def predict(self):
        return self.lr_model.predict(self.scaler.transform(self.datasets["X_test"]))
    
    def validate(self):
        return self.lr_model.predict(self.scaler.transform(self.datasets["X_validate"]))
    