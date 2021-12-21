from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, concatenate
from keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard

import keras_tuner as kt

import joblib
import numpy as np
from utils import img_scaler, scaler

np.set_printoptions(suppress=True, precision=5)


def build_optimized_model(hp):

    input1 = Input((20, 69, 1))
    
    x1 = Conv2D(hp.Choice('filters1', [32, 64, 96]), 3, activation='relu', padding='same')(input1)
    if hp.Boolean('batchnorm1'):
        x1 = BatchNormalization()(x1)
    else:
        x1 = Dropout(hp.Choice('dropout1', [0.0, 0.1, 0.2]))(x1)
    x1 = Conv2D(hp.Choice('filters2', [32, 64, 96]), 3, activation='relu', padding='same')(x1)
    x1 = MaxPooling2D(pool_size=(3,3))(x1)
    
    x1 = Dropout(hp.Choice('dropout2', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], default=0.2))(x1)

    x1 = Conv2D(hp.Choice('filters3', [64, 96, 128, 192]), 3, activation='relu', padding='same')(x1)
    if hp.Boolean('batchnorm2'):
        x1 = BatchNormalization()(x1)
    else:
        x1 = Dropout(hp.Choice('dropout3', [0.0, 0.1, 0.2]))(x1)
    x1 = Conv2D(hp.Choice('filters4', [64, 96, 128, 192], default=128), 3, activation='relu', padding='same')(x1)
    x1 = MaxPooling2D(pool_size=(3,3))(x1)

    flat1 = Flatten()(x1)
    flat1 = Dropout(hp.Float('dropout4', min_value=0.2, max_value=0.75, sampling='linear', default=0.5))(flat1)
    
    input2 = Input((256,))
    input3 = Input((3,))

    x2 = concatenate([input2, input3])

    x2 = Dense(hp.Int('units1', min_value=32, max_value=256, step=32, default=256), activation='relu')(x2)
    flat2 = Dropout(hp.Choice('dropout5', [0.0, 0.1, 0.2, 0.3, 0.4], default=0.2))(x2)
  
    merge = concatenate([flat1, flat2])
    
    x = Dense(hp.Int('units2', min_value=64, max_value=512, step=64, default=256), activation='relu')(merge)

    x = Dropout(hp.Float('dropout6', min_value=0.2, max_value=0.75, sampling='linear', default=0.5))(x)
    
    output = Dense(8, activation='softmax')(x)
    
    model = Model(inputs=[input1, input2, input3], outputs=output, name='OptimizedConv2D')
                    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
    return model

if __name__ == '__main__':

    X_train_keras, X_val_keras = joblib.load('data/X_train_keras'), joblib.load('data/X_val_keras')
    X_keras, X_test_keras = joblib.load('data/X_keras'), joblib.load('data/X_test_keras')

    y_train_keras, y_val_keras = joblib.load('data/y_train_keras'), joblib.load('data/y_val_keras')

    X_train_keras[1], (X_val_keras[1], X_test_keras[1]) = scaler(X_train_keras[1], [X_val_keras[1], X_test_keras[1]])
    X_train_keras[0], (X_val_keras[0], X_test_keras[0]) = img_scaler(X_train_keras[0], [X_val_keras[0], X_test_keras[0]])

            
    tuner = kt.BayesianOptimization(build_optimized_model, objective='val_loss', max_trials=250, num_initial_points=80,                                 
                                    directory = 'keras/search')
    

    def scheduler(epoch, lr):
        if epoch == 50 or epoch == 70 or epoch == 90:
            return lr*0.5
        else:
            return lr

    callbacks = [EarlyStopping(patience=30, min_delta=0.0001),
                 LearningRateScheduler(scheduler),
                 TensorBoard(log_dir='keras/tb_logs')]


    tuner.search([X_train_keras[0], X_train_keras[1], X_train_keras[2]], y_train_keras,
                  validation_data=([X_val_keras[0], X_val_keras[1], X_val_keras[2]], y_val_keras),
                  epochs=200, batch_size=8, callbacks=callbacks)