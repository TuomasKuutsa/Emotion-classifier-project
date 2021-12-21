import numpy as np
import pandas as pd
from time import time
from utils import time_taken

import joblib

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedShuffleSplit


def do_incrementalsearch(X, y, estimator, grid, resources, splits, scoring, n_jobs, verbose):

    """
    Function implements hyperparameter search in a partial manner. User can spesify how much data is used in each iteration.
    First iteration is always a RandomSearch and possible later iterations use user specified amount of best performing hyperparameters with specified amount of data.

    X:          Training data array.
    y:          traget array.
    estimator:  Sklearn estimator.
    recources:  list of tuples specifying how much resources are used in each iteration. e.g. if len(X) = 10 000, [(3000, 1000), (6000, 200), (10 000, 20)],
                then search would use 3000 samples out of 10000 and sample 1000 hyperparameters in the first iteration, 6000 samples and 200 best performing hps on the second
                and all of available data and 20 best performing hps on the last iteration.
    splits:     list of sklearn fold objects for each iteration.
    n_jobs:     Number or parallel jobs.
    verbose:    Degree of verbosity of search

    """
    
    searches = []
    results_list = []
    
    for i , ((n_samples, n_candidates), cv) in enumerate(zip(resources, splits)):
        
        if i == 0:
            
            print('\n========== Started RandomizedSearch ==============\n') 
            print(f'Iteration step: {i+1}/{len(resources)}')                
            print(f'Using {n_samples} samples with {n_candidates} random parameter candidates\n')
            print('==================================================\n')
            
            X_incremental , y_incremental = [], []

            if n_samples == len(y):
                X_incremental = X
                y_incremental = y
            else:
                idx, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=n_samples).split(X,y))
                X_incremental, y_incremental = X[idx], y[idx]
            
            
            search = RandomizedSearchCV(estimator=estimator, param_distributions=grid,
                                            n_iter=n_candidates, cv=cv, scoring=scoring,
                                            n_jobs=n_jobs, verbose=verbose)

            start = time()
            search.fit(X_incremental, y_incremental)    
            print(time_taken(start, True))

            results = search.cv_results_
            results['n_samples'] = [n_samples]*n_candidates
            
            results_list.append(pd.DataFrame(results))
            searches.append(search)
            
            joblib.dump(searches, 'searches/tempsearches'), joblib.dump(pd.concat(results_list), 'searches/tempresults')
            
        else:            
            print('\n========== Started GridSearch ==============\n')
            print(f'Iteration step: {i+1} / {len(resources)}')
            print(f'Using {n_samples} samples with {n_candidates} best parameter candidates\n')
            print('============================================\n')
            
            
            if n_samples == len(y):
                
                grid = get_params(searches[-1].cv_results_, n_candidates, True)

                search = GridSearchCV(estimator=estimator, param_grid=grid,
                                      cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)

                start = time()                   
                search.fit(X, y)  
                print(time_taken(start, True))

                results = search.cv_results_
                results['n_samples'] = [n_samples]*n_candidates
                results_list.append(pd.DataFrame(results))
                
                searches.append(search)

                get_params(searches[-1].cv_results_, 3, False)
                
                joblib.dump(searches, 'searches/tempsearches'), joblib.dump(pd.concat(results_list), 'searches/tempresults')
                
                return searches, pd.concat(results_list)
            
            else:
                
                idx, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=n_samples).split(X,y))
                X_incremental, y_incremental = X[idx], y[idx]
                
                grid = get_params(searches[-1].cv_results_, n_candidates, True)

               
                search = GridSearchCV(estimator=estimator, param_grid=grid,
                                      cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)

                start = time()
                search.fit(X_incremental, y_incremental)
                print(time_taken(start, True))

                results = search.cv_results_
                results['n_samples'] = [n_samples]*n_candidates
                results_list.append(pd.DataFrame(results))
                
                searches.append(search)
                
                joblib.dump(searches, 'searches/tempsearches'), joblib.dump(pd.concat(results_list), 'searches/tempresults')
                
    get_params(searches[-1].cv_results_, 3, False)
    
    joblib.dump(searches, 'searches/searches'), joblib.dump(pd.concat(results_list), 'searches/results')

    return searches, pd.concat(results_list)

            
def get_params(results, k, wrap_values):

    params, rank = (np.asarray(a) for a in (results['params'], results['rank_test_score']))
    sorted_params = params[np.argsort(rank)][:k]
    print(f'Top parameters:\n\n {sorted_params[:3]}\n')
    print('==================== END =======================\n')
    
    if wrap_values:
        return [{k:[v] for (k,v) in p.items()} for p in sorted_params]
    return None

def get_reference(results_frame):    
    
    samples = np.flip(results_frame['n_samples'].unique())

    sorted_results = results_frame[['mean_test_score','n_samples', 'params']].sort_values('mean_test_score', ascending=False)

    sorted_params_last = [p for p in sorted_results[sorted_results['n_samples']==samples[0]]['params'].tolist()]

    reference = []

    for i, n_sample in enumerate(samples[1:]):
        sorted_params_previous_iteration = [p for p in sorted_results[sorted_results['n_samples']==n_sample]['params'].tolist()]

        if i == 0:
            for j, p1 in enumerate(sorted_params_last):
                for k, p2 in enumerate(sorted_params_previous_iteration):
                    if p1 == p2:
                        reference.append([j+1, k+1])

        else:
            for j, p1 in enumerate(sorted_params_last):
                for k, p2 in enumerate(sorted_params_previous_iteration):
                    if p1 == p2:
                        reference[j].append(k+1)
    
    return reference