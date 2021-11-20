import traceback
import numpy as np
import time

import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from IncrementalSearch import incremental_search, time_taken
import scipy.stats as sst

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split



data, target = joblib.load('data/data'), joblib.load('data/target')

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)


pipe_lr = Pipeline([
                    ('reduce', RFE(estimator=RandomForestClassifier())),
                    ('scaler', StandardScaler()),
                    ('clf', LogisticRegression(max_iter=200))   
                    ])

pipe_svc = Pipeline([
                    ('reduce', RFE(estimator=RandomForestClassifier())),
                    ('scaler', StandardScaler()),
                    ('clf', SVC())   
                    ])

pipe_sgd = Pipeline([
                    ('reduce', RFE(estimator=RandomForestClassifier())),
                    ('scaler', None),
                    ('clf', SGDClassifier(max_iter=1500))
                    ])

pipe_rfc = Pipeline([
                    ('reduce', RFE(estimator=RandomForestClassifier())),
                    ('scaler', None),
                    ('clf', RandomForestClassifier())   
                    ])

pipe_xgb = Pipeline([
                    ('reduce', RFE(estimator=RandomForestClassifier())),
                    ('scaler', None),
                    ('clf', XGBClassifier(use_label_encoder=False)) 
                    ])



params_lr = {   
                'reduce__n_features_to_select':[4000],
                'reduce__step':[0.1],

                'clf__solver': ['liblinear'],
                'clf__penalty': ['l1', 'l2'],
                'clf__C': sst.loguniform(0.0001, 10)
            }


params_svc = [  {
                    'reduce__n_features_to_select':[4000],
                    'reduce__step':[0.1],

                    'clf__kernel': ['poly'],
                    'clf__C': sst.loguniform(0.001, 10),
                    'clf__gamma': sst.loguniform(0.00001, 1),              
                    'clf__degree': [2,3,4,5]
                },

                {
                    'reduce__n_features_to_select':[4000],
                    'reduce__step':[0.1],

                    'clf__C': sst.loguniform(0.001, 10),
                    'clf__gamma': sst.loguniform(0.0001, 1),
                    'clf__kernel': ['rbf', 'sigmoid', 'linear']
                }
            ]

params_sgd =   [
                    {   
                        'reduce__n_features_to_select':[1000],
                        'reduce__step':[0.2, 0.1],
                        'scaler': [StandardScaler(), MinMaxScaler()],

                        'clf__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                        'clf__alpha': sst.loguniform(0.0001, 1),
                        'clf__eta0': sst.loguniform(0.0001, 1),
                        'clf__learning_rate': ['constant', 'adaptive', 'invscaling']
                    },
    
                    {   
                        'reduce__n_features_to_select':[1000],
                        'reduce__step':[0.2, 0.1],
                        'scaler': [StandardScaler(), MinMaxScaler()],

                        'clf__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                        'clf__alpha': sst.loguniform(0.0001, 10),
                        'clf__learning_rate': ['optimal']
                    }
                ]

params_rfc =    {
                    'reduce__n_features_to_select':[1000],
                    'reduce__step':[0.2, 0.1],
                    'scaler': [None, StandardScaler(), MinMaxScaler()],

                    'clf__max_depth': np.arange(1, 21, 1),
                    'clf__max_features': np.arange(5, 50, 1),
                    'clf__n_estimators': np.arange(25, 500, 25),
                    'clf__max_samples': sst.uniform(0.5, 0.5)
                }

params_xgb =    {   
                    'reduce__n_features_to_select':[1000],
                    'reduce__step':[0.2, 0.1],
                    'scaler': [None, StandardScaler(), MinMaxScaler()],

                    'clf__tree_method': ['gpu_hist'],
                    'clf__eval_metric': ['mlogloss'],
                    'clf__objective': ['multi:softmax'],
                    'clf__n_estimators': np.arange(50, 300, 10),
                    'clf__min_child_weight':np.arange(1, 6, 1),
                    'clf__max_depth': np.arange(1, 11, 1),
                    'clf__learning_rate': sst.uniform(0.01, 0.4),
                    'clf__subsample': sst.uniform(0.3, 0.7),
                    'clf__gamma': sst.loguniform(0.000001, 1),
                    'clf__sampling_method': ['uniform', 'gradient_based'],
                    'clf__grow_policy': ['depthwise', 'lossguide']             
                }

    


steps = [(1000, 500),  (2000, 100), (3456, 20)]
# steps = [(50, 10), (60, 5),  (70, 2)]

# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
split = StratifiedKFold(n_splits=2)

paths = [('searches/'+m+'Search', 'searches/'+m+'Results') for m in ['LR', 'SVC']]

pipes = [pipe_lr, pipe_svc]
grids = [params_lr, params_svc]

t1 = time()

for pipe, grid, path in zip(pipes, grids, paths):

    try:
    
        t2 = time()
        inc_search, results = incremental_search(X=X_train, y=y_train, estimator=pipe, grid=grid, search_steps=steps, cv=split, scoring='f1_macro', n_jobs=-1, verbose=100)

        joblib.dump(inc_search, path[0]), joblib.dump(results, path[1])

        time_taken(t2, True)

    except Exception:
        print(traceback.format_exc())

inc_search, results = incremental_search(X=X_train, y=y_train, estimator=pipe_xgb, grid=params_xgb, search_steps=steps, cv=split, scoring='f1_macro', n_jobs=6, verbose=10)

joblib.dump(inc_search, 'searches/XGBCSearch'), joblib.dump(results, 'searches/XGBCResults')
    
time_taken(t1, True)

