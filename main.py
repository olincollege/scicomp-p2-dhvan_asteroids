import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time

RELOAD_DATA = False

if(RELOAD_DATA):
    print("RELOADING RAW DATA")
    _time = time.perf_counter()
    complete_asteroid_dataframe = pd.read_csv('raw_data/asteroids.csv', skiprows=2, sep=r'[\s;]+', engine='python')
    families_dataframe = pd.read_csv('raw_data/families.csv', sep=r'\s+', engine='python')
    
    complete_asteroid_dataframe['no'] = complete_asteroid_dataframe['no'].astype(str)
    families_dataframe['%ast.name'] = families_dataframe['%ast.name'].astype(str)
    
    raw_merged_df = pd.merge(complete_asteroid_dataframe, families_dataframe, left_on="no", right_on="%ast.name", how="inner")
    dataset = raw_merged_df[['a', 'ecc', 'sinI', 'family1']]
    
    _dataset, test_dataset = train_test_split(dataset, train_size=0.8, test_size=0.2)
    train_dataset, validate_dataset = train_test_split(_dataset, train_size=0.8, test_size=0.2)
    
    with open('saved_obj/complete_asteroid_dataframe.pkl', 'wb') as f:
        pickle.dump(complete_asteroid_dataframe, f)
        
    with open('saved_obj/families_dataframe.pkl', 'wb') as f:
        pickle.dump(families_dataframe, f)
        
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
    with open('saved_obj/complete_asteroid_dataframe.pkl', 'rb') as f:
        complete_asteroid_dataframe = pickle.load(f)
    
    with open('saved_obj/families_dataframe.pkl', 'rb') as f:
        families_dataframe = pickle.load(f)
        
    with open('saved_obj/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
        
    with open('saved_obj/train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
        
    with open('saved_obj/test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
        
    with open('saved_obj/validate_dataset.pkl', 'rb') as f:
        validate_dataset = pickle.load(f)
    print(f'Saved files loaded in {time.perf_counter() - _time}s')
    

truncate = -1
X_train = train_dataset[['a', 'ecc', 'sinI']].iloc[0:truncate]
Y_train = train_dataset['family1'].iloc[0:truncate]

X_test = test_dataset[['a', 'ecc', 'sinI']].iloc[0:truncate]
Y_test = test_dataset['family1'].iloc[0:truncate]

scaler = StandardScaler()
_time = time.perf_counter()
model = LogisticRegression(max_iter=1000)
model.fit(scaler.fit_transform(X_train), Y_train)
print(f'Model trained in {time.perf_counter() - _time}s')

_time = time.perf_counter()
predictions = model.predict(scaler.transform(X_test))
print(f'Predicions in {time.perf_counter() - _time}s')
accuracy = accuracy_score(Y_test, predictions)
print(f"Accuracy Score = {accuracy * 100:.2f}%")