from typing import Self

import pandas as pd
from pandas import DataFrame as df
from algorithms.algorithm import Algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from decorators import timer

class AsteroidLogisticRegression(Algorithm):
    
    X_train: df
    Y_train: df
    
    X_test: df
    Y_test: df
    
    X_validate: df
    Y_validate: df

    def __init__(self, reload_raw_data = False, algorithm_name = "LogisticRegression"):
        super().__init__(reload_raw_data, algorithm_name)
        self.scaler = StandardScaler()
        self.lr_model = LogisticRegression(max_iter=1000)
        self.fit()
        
    @timer
    def fit(self):
        return self.lr_model.fit(self.scaler.fit_transform(self.X_train), self.Y_train)
    
    def predict(self):
        return self.lr_model.predict(self.scaler.transform(self.X_test))
    
    def validate(self):
        return self.lr_model.predict(self.scaler.transform(self.X_validate))
    