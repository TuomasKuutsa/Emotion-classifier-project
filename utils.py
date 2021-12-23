import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn. model_selection import StratifiedKFold
from IPython.display import clear_output
from tensorflow import keras
import keras_tuner as kt
from multiprocessing import Pool
from process_data import get_processed


class PlotMetrics(keras.callbacks.Callback):

    """
    Callback class for plotting perfomance metrics in real-time when training a keras model.

    """
     
    def __init__(self):
        self.losses = []
        self.accs = []
        self.val_losses = []
        self.val_accs = []
        self.limit = 50
        self.step = 5
        
    def on_epoch_end(self, epoch, logs=None):
        
        fig, axs = plt.subplots(1,2, figsize=(12,6))
        
        loss, val_loss, acc, val_acc = logs['loss'], logs['val_loss'], logs['accuracy'], logs['val_accuracy']
        
        self.losses.append(loss)
        self.val_losses.append(val_loss)
        self.accs.append(acc)
        self.val_accs.append(val_acc)
        
        N = range(epoch+1)
        
        if epoch+1 == self.limit:
            self.step = 2*self.step
            self.limit = 2*self.limit
        if epoch+1 <= 50:
            x_ticks = np.arange(0, self.limit+1, self.step)
        else:
            x_ticks = np.arange(0, epoch+1, self.step)
        
        clear_output(wait=True)
        axs[0].plot(N, self.losses, 'r', label='Training loss')
        axs[0].plot(N, self.val_losses, 'b', label='Validation loss')
        axs[0].grid()
        axs[0].legend()
        axs[0].set_ylim(0, 2)
        axs[0].yaxis.set_ticks(np.arange(0, 2.7, 0.2))
        axs[0].xaxis.set_ticks(x_ticks)
        axs[0].set_xlabel('Epochs'), axs[0].set_ylabel('Validation loss')
        axs[0].set_title('Loss over epochs')

        axs[1].plot(N, self.accs, 'r', label='Training accuracy')
        axs[1].plot(N, self.val_accs, 'b', label='Validation accuracy')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_ylim(0, 1.1)
        axs[1].yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        axs[1].xaxis.set_ticks(x_ticks)
        axs[1].set_xlabel('Epochs'), axs[1].set_ylabel('Validation accuracy')
        axs[1].set_title('Accuracy over epochs')

        plt.show()

class KerasModelProgress(keras.callbacks.Callback):

    """
    Callback class for monitoring keras model fitting.

    max_epochs:         number of epochs you are fitting a model.
    update_step (int):  Number of update messages when fitting.
    time:               time in seconds before calling model.fit for monitoring time passed.

    """

    def __init__(self, max_epochs, n_updates, time):
        self.max_epochs = max_epochs
        self.n_updates = n_updates
        self.time = time
             
    def on_epoch_begin(self, epoch, logs=None):       
        if epoch+1 == 1 or (epoch+1) % ceil(self.max_epochs/self.n_updates) == 0:
            print(f'Model fitting on epoch {epoch+1}/{self.max_epochs}, {round(((epoch+1)/self.max_epochs)*100, 2)}%, has taken {time_taken(self.time, True)}')

        
def plot_metrics(history):

    """
    Plotting function for keras history object.
    Plots loss, val_loss, accuracy and val_accuracy

    """
    
    fitted_epochs = len(history['val_loss'])
    
    print(f"Lowest validation loss on epoch: {np.argmin(history['val_loss']) + 1}\n")

    fig = plt.figure(figsize=(12,6))

    ax = fig.add_subplot(121)
    ax.plot(range(fitted_epochs), history['loss'], 'r', label='Training loss')
    ax.plot(range(fitted_epochs), history['val_loss'], 'b', label='Validation loss')
    ax.grid()
    ax.legend()
    ax.set_ylim(0, 2)
    ax.yaxis.set_ticks(np.arange(0, 2.1, 0.2))
    ax.set_xlabel('Epochs'), ax.set_ylabel('Validation loss')
    ax.set_title('Loss over epochs')

    ax = fig.add_subplot(122)
    ax.plot(range(fitted_epochs), history['accuracy'], 'r', label='Training accuracy')
    ax.plot(range(fitted_epochs), history['val_accuracy'], 'b', label='Validation accuracy')
    ax.grid()
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('Epochs'), ax.set_ylabel('Validation accuracy')
    ax.set_title('Accuracy over epochs')

    plt.show()



def img_scaler(train_set, test_set=None):
    
    """
    Standardizes an array of shape (n_samples, n_rows, n_columns, channels) using StandardScaler.
    If test_set is provided it is transformed using StandardScaler that was fitten on the train set.

    train_set:  Array
    test_set:   Array or list of arrays

    """       
    ss = StandardScaler()    
    train_set = ss.fit_transform(train_set.reshape(-1, train_set[0].flatten().shape[0])).reshape(train_set.shape)
    if isinstance(test_set, np.ndarray):
        return train_set, ss.transform(test_set.reshape(-1, test_set[0].flatten().shape[0])).reshape(test_set.shape)
    elif isinstance(test_set, list):
        return train_set, (ss.transform(s.reshape(-1, s[0].flatten().shape[0])).reshape(s.shape) for s in test_set)
    else:
        return train_set

def scaler(train_set, test_set=[]):

    """
    Standardizes an array of shape (n_samples, n_columns) using StandardScaler.
    If test_set is provided it is transformed using StandardScaler that was fitted on the train set.

    train_set:  Array
    test_set:   Array or list of arrays

    """

    ss = StandardScaler()    
    train_set = ss.fit_transform(train_set)
    if isinstance(test_set, np.ndarray):
        return train_set, ss.transform(test_set)
    elif isinstance(test_set, list):
        return train_set, (ss.transform(s) for s in test_set)
    else:
        return train_set

def time_taken(s, as_start_time):

    """
    Formatter from seconds to hh:mm:ss.
    if as_start_time True calculates time difference relative to input time.

    s (int):        Seconds. Use for example time.time()              
    as_start_time:  Boolean
    """

    if as_start_time:
        m, s = divmod(time()-s, 60)
        h, m = divmod(m, 60)
        return f"{int(h)}h{int(m)}min{int(s)}s"
    else:
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{int(h)}h{int(m)}min{int(s)}s"


# =================================================================================

def sklearn_cv(clf_list, train_paths, n_folds, n_augmentations, n_parallel):

    """
    Cross-valdiation implementation for sklearn estimators.
    If augmentation is used then only the training fold only and predicts and scores using valdiation fold.

    clf_list:           list of Sklearn objects that implement estimator api i.e classifiers, pipelines, searches etc.
    train_paths:        paths to raw data
    n_folds: int,       number of (stratified)folds
    n_augmentations:    int, number of augmentations in the training fold

    returns a score dictionary

    """
                
    classifier_scores = {}
    
    for clf in clf_list:
        
        print(f'Classifier: {clf[-1].__class__.__name__}\n')
        
        folds = get_folds(train_paths, n_augmentations, n_folds)
        
        args = [(fold[0], fold[1], fold[2], clf) for fold in folds]
        
        scores = get_cv_scores(args, n_parallel)
        
        print()
        
        classifier_scores[(clf.__class__.__name__, n_augmentations)] = scores

    return classifier_scores

def get_folds(train_paths, n_augmentations, n_folds, for_sklearn=True):

    """
    Function creates cross-validation folds from train_paths.

    returns list of length n_folds with elements of shape (fold number, (X_train, y_train), (X_val, y_val))

    """
    
    l = []

    n_mels = 64
    if not for_sklearn:
        n_mels=128
    
    folds = StratifiedKFold(n_splits=n_folds)
    for i, (train, val) in enumerate(folds.split(train_paths, [int(path.split('-')[6]) for path in train_paths])):

        t = time()
        X_train, y_train = get_processed(np.array(train_paths)[train], n_augmentations=n_augmentations, for_sklearn=for_sklearn, n_mels=n_mels)
        X_val, y_val = get_processed(np.array(train_paths)[val], for_sklearn=for_sklearn, n_mels=n_mels)
        print(f'Fold {i+1} data processing took {time_taken(t, True)}')
        
        l.append([i+1, (X_train, y_train), (X_val, y_val)])

    return l

def process(fold, train_set, val_set, clf):

    """
    Fitting and scoring function for multiprocessing.

    """
    
    print(f'Fitting fold {fold}')
    t = time()
    clf.fit(train_set[0], train_set[1])
    print(f'finished fold {fold}, took {time_taken(t, True)}')
    
    return f1_score(val_set[1], clf.predict(val_set[0]), average='macro')

def get_cv_scores(args, n_parallel):

    """
    Multiprocessing implementation for fitting and scoring CV folds created by get_folds.

    Returns an array of fold scores.

    """       
    scores = []

    with Pool(processes=n_parallel) as pool:
        processed = pool.starmap(process, args)
        for score in processed:
            scores.append(score)   
       
    return np.asarray(scores)

# ============================================================================================


def keras_cv(build_model, hp, train_paths, n_augmentations, n_folds, epochs, n_msgs):

    """
    Cross-validation for keras models.

    build_model:        keras model.
    hp:                 Possible hyperparameters object to build model or None.
    train_paths:        Paths to raw data.
    n_augmentations:    Number of augmentations in traning fold.
    epochs:             Number of epochs to train the model.
    n_msgs:             Number of traning progress messages.

    """
    
    def scheduler(epoch, lr):
        if epoch == 50 or epoch == 70 or epoch == 90:
            return lr*0.5
        else:
            return lr
    
    folds = get_folds(train_paths, n_augmentations, n_folds, for_sklearn=False)
    
    f1s = []
    losses = []
    
    for fold in folds:
        
        n_fold = fold[0]
        X_train, y_train = fold[1][0], fold[1][1]
        X_val, y_val = fold[2][0], fold[2][1]      
        
        X_train[1], X_val[1] = scaler(X_train[1], X_val[1])
        
        y_train = keras.utils.to_categorical(y_train, 8)
        y_val_cat = keras.utils.to_categorical(y_val, 8)
        
        if isinstance(hp, kt.HyperParameters):    
            model = build_model(hp)
        else:
            model= build_model()
        
        print(f'\nFitting fold {n_fold}')
              
        t = time()
        progress = KerasModelProgress(epochs, n_msgs, t)
        model.fit([X_train[0], X_train[1], X_train[2]], y_train, batch_size=8, epochs=epochs, verbose=0,
                   callbacks=[keras.callbacks.LearningRateScheduler(schedule=scheduler),
                              progress])
        print(f'Fold fitted, took time {time_taken(t, True)}')
        
                       
        y_pred = np.argmax(model.predict([X_val[0], X_val[1], X_val[2]]), axis=-1)
        f1 = f1_score(y_val, y_pred, average='macro')
               
        logs = model.evaluate([X_val[0], X_val[1], X_val[2]], y_val_cat, batch_size=8)
        
        f1s.append(f1)
        losses.append(logs[0])
        
    return f1s, losses

      


