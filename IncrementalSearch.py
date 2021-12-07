import numpy as np
import pandas as pd
from time import time

import joblib

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedShuffleSplit


def incremental_search(X, y, estimator, grid, search_steps, splits, scoring, n_jobs, verbose):
    
    searches = []
    results_list = []
    
    for i , ((n_samples, n_candidates), cv) in enumerate(zip(search_steps, splits)):
        
        if i == 0:
            
            print('\n========== Started RandomizedSearch ==============\n') 
            print(f'Iteration step: {i+1}/{len(search_steps)}')                
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
            time_taken(start, True)

            results = search.cv_results_
            results['n_samples'] = [n_samples]*n_candidates
            
            results_list.append(pd.DataFrame(results))
            searches.append(search)
            
            joblib.dump(searches, 'searches/tempsearches'), joblib.dump(pd.concat(results_list), 'searches/tempresults')
            
        else:            
            print('\n========== Started GridSearch ==============\n')
            print(f'Iteration step: {i+1} / {len(search_steps)}')
            print(f'Using {n_samples} samples with {n_candidates} best parameter candidates\n')
            print('============================================\n')
            
            
            if n_samples == len(y):
                
                grid = get_params(searches[-1].cv_results_, n_candidates, True)

                search = GridSearchCV(estimator=estimator, param_grid=grid,
                                      cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)

                start = time()                   
                search.fit(X, y)  
                time_taken(start, True)

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
                time_taken(start, True)

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


def time_taken(s, as_start_time):

    if as_start_time:
        m, s = divmod(time()-s, 60)
        h, m = divmod(m, 60)
        print(f"\nTook time: {int(h)}h{int(m)}min{int(s)}s\n")
    else:
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{int(h)}h{int(m)}min{int(s)}s"

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