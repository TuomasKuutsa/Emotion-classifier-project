from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D

import keras_tuner as kt

import joblib

X_train_keras, X_val_keras, y_train_keras, y_val_keras = joblib.load('data/X_train_keras'), joblib.load('data/X_val_keras'), joblib.load('data/y_train_keras'), joblib.load('data/y_val_keras')

y_train_cat = keras.utils.to_categorical(y_train_keras, 8)
y_val_cat = keras.utils.to_categorical(y_val_keras, 8)

print(X_train_keras[0].shape)


def build_model(hp):
    model = Sequential()
    
    model.add(Input(shape=X_train_keras[1].shape))

    model.add(Conv2D(
        hp.Int('filters1', min_value=16, max_value=64, step=16, default=32), 
        kernel_size=3,
        activation='relu',
        padding='same'))

    model.add(Dropout(hp.Choice('dropout1', [0.0, 0.2, 0.3, 0.4])))

    model.add(Conv2D(
        hp.Int('filters2', min_value=16, max_value=64, step=16, default=32),
        kernel_size=3,
        activation='relu',
        padding='same'))
    model.add(MaxPooling2D(pool_size=(3,3)))

    model.add(Dropout(hp.Float('dropout2', min_value=0.2, max_value=0.75, sampling='linear')))

    model.add(Conv2D(
        hp.Int('filters3', min_value=32, max_value=128, step=16, default=64), 
        kernel_size=3,
        activation='relu',
        padding='same'))
    
    model.add(Dropout(hp.Choice('dropout3', [0.0, 0.2, 0.3, 0.4])))

    model.add(Conv2D(
        hp.Int('filters4', min_value=32, max_value=128, step=16, default=64), 
        kernel_size=3,
        activation='relu',
        padding='same'))
    model.add(MaxPooling2D(pool_size=(3,3)))

    model.add(Flatten())

    model.add(Dropout(hp.Float('dropout4', min_value=0.2, max_value=0.75, sampling='linear')))
              
    model.add(Dense(
        units = hp.Int('units1', min_value=64, max_value=512, step=64),
        activation = 'relu'))

    model.add(Dropout(hp.Float('dropout5', min_value=0.2, max_value=0.5, sampling='linear')))

    model.add(Dense(8, activation='softmax'))
    
    model.compile(keras.optimizers.Adam(0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        
    return model
        

tuner = kt.BayesianOptimization(build_model, objective='val_loss', max_trials=200, num_initial_points=80,                                 
                                directory = 'keras/search')

callbacks = [keras.callbacks.EarlyStopping(patience=20),
             keras.callbacks.ReduceLROnPlateau(factor=0.7, patience=5),
             keras.callbacks.TensorBoard(log_dir='keras/tb_logs')]


tuner.search(X_train_keras, y_train_cat, validation_data=(X_val_keras, y_val_cat), epochs=120, batch_size=16, callbacks=callbacks)

best_hps = tuner.get_best_hyperparameters(5)

joblib.dump(best_hps, 'searches/KerasResults')