import traceback
import numpy as np
from time import time

import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from IncrementalSearch import incremental_search, time_taken
import scipy.stats as sst

from sklearn.model_selection import StratifiedKFold



X_train, y_train = joblib.load('data/X_train'), joblib.load('data/y_train')

pipe_lr = Pipeline([
                    ('scaler', None),
                    ('clf', LogisticRegression(max_iter=2000))   
                    ])

pipe_svc = Pipeline([
                    ('scaler1', StandardScaler()),
                    ('decomp', None),
                    ('scaler2', None),
                    ('clf', SVC())   
                    ])


pipe_rfc = Pipeline([
                    ('clf', RandomForestClassifier())   
                    ])

pipe_mlp = Pipeline([
                    ('scaler', None),
                    ('clf', MLPClassifier())   
                    ])

pipe_xgb = Pipeline([
                    ('clf', XGBClassifier(use_label_encoder=False, n_jobs=1)) 
                    ])



params_lr = [{   
                'scaler': [StandardScaler(), MinMaxScaler()],

                'clf__solver': ['liblinear'],
                'clf__penalty': ['l1', 'l2'],
                'clf__class_weight': [None, 'balanced'],
                'clf__C': sst.loguniform(0.001, 10)
            },

            {
                'scaler': [StandardScaler(), MinMaxScaler()],

                'clf__solver': ['lbfgs'],
                'clf__penalty': ['l2'],
                'clf__class_weight': [None, 'balanced'],
                'clf__C': sst.loguniform(0.001, 10)
            }]


params_svc = [  
                {
                    'decomp': ['passthrough', PCA(), PCA(0.99)],
                    'scaler2': ['passthrough', StandardScaler(), MinMaxScaler()],

                    'clf__kernel': ['poly'],
                    'clf__C': sst.loguniform(0.001, 10),
                    'clf__gamma': sst.loguniform(0.00001, 10),              
                    'clf__degree': [2,3,4,5]
                },

                {
                    'decomp': ['passthrough', PCA(), PCA(0.99)],
                    'scaler2': ['passthrough', StandardScaler(), MinMaxScaler()],

                    'clf__C': sst.loguniform(0.001, 10),
                    'clf__gamma': sst.loguniform(0.00001, 10),
                    'clf__kernel': ['rbf']
                },

                {
                    'decomp': ['passthrough', PCA(), PCA(0.99)],
                    'scaler2': ['passthrough', StandardScaler(), MinMaxScaler()],

                    'clf__C': sst.loguniform(0.001, 10),
                    'clf__gamma': sst.loguniform(0.00001, 10),
                    'clf__kernel': ['linear']
                }
            ]

params_rfc =    {   
                    'clf__n_estimators': [500],
                    'clf__criterion': ['gini', 'entropy'],
                    'clf__max_features': np.arange(5, 50, 1),
                    'clf__min_samples_split': np.arange(2, 11, 1),
                    'clf__bootstrap': [True, False]

                }

params_mlp =    {   
                    'scaler': [StandardScaler(), MinMaxScaler()],

                    'clf__max_iter': [1000],
                    'clf__activation': ['relu', 'tanh'],
                    'clf__batch_size': [200, 100, 50],
                    'clf__learning_rate_init': [0.001, 0.0005],
                    'clf__learning_rate': ['constant'],
                    'clf__hidden_layer_sizes': [(512,), (256,), (128,), (64,), (512, 128), (512, 64),
                                                (256, 128), (256, 64), (128, 128), (128, 64), (64, 64), (64, 32)]
                },

params_xgb =    {   
                    'clf__tree_method': ['gpu_hist'],
                    'clf__eval_metric': ['mlogloss'],
                    'clf__objective': ['multi:softmax'],
                    'clf__n_estimators': [500],
                    'clf__min_child_weight': [2, 3, 4, 5, 6],
                    'clf__max_depth': [3, 4, 5, 6],
                    'clf__learning_rate': [0.05, 0.1, 0.3],
                    'clf__subsample': sst.uniform(0.5, 0.5),
                    'clf__colsample_bytree': sst.uniform(0.5, 0.5),
                    'clf__gamma': sst.loguniform(0.0001, 0.1)
            
                }


steps = [(len(y_train), 1500)]

paths = [('searches/'+m+'Search', 'searches/'+m+'Results') for m in ['RandomForestClassifier']]

pipes = [pipe_rfc]
grids = [params_rfc]
splits = [StratifiedKFold(n_splits=5)]

t1 = time()

print(paths)

for pipe, grid, path in zip(pipes, grids, paths): 

    try:
    
        t2 = time()
        search, results = incremental_search(X=X_train, y=y_train, estimator=pipe, grid=grid, search_steps=steps, splits=splits, scoring='f1_macro', n_jobs=-1, verbose=10)

        joblib.dump(search, path[0]), joblib.dump(results, path[1])

        time_taken(t2, True)

    except Exception:
        print(traceback.format_exc())

