import numpy as np
import pandas as pd
from tensorflow.keras import *


def create_model(frames, matrix_rows, matrix_cols, channels):
    """
    Generate model capable of handling multi-channel EEG time-series
    """
    _model = models.Sequential(name='ASD Classifier')
    _model.add(layers.Input(shape=(frames, matrix_rows, matrix_cols, channels), name='eeg_signals'))
    _model.add(layers.TimeDistributed(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu'), name='conv'))
    _model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)), name='pooling'))
    _model.add(layers.TimeDistributed(layers.Flatten(), name='flatten'))
    _model.add(layers.LSTM(32, dropout=0.2))
    _model.add(layers.Dense(1, activation='sigmoid'))
    _model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return _model


if __name__ == '__main__':
    df = pd.read_feather('data/dataset-clean.ftr')
    participants = df['Participant'].unique()
    epochs = df['Epoch'].unique()[1:]  # ignore baseline

    TIME_SLICE = 12000

    dataset = np.zeros(shape=(len(participants) * len(epochs), TIME_SLICE, 5, 6, 1))

    # transform dataset into trainable form
    df = df.set_index('Participant')
    i = 0
    for c in participants:
        p = df.loc[c].set_index('Epoch')
        for e in epochs:
            dataset[i] = np.expand_dims(p.loc[e].set_index('T').iloc[:TIME_SLICE].to_numpy().reshape((TIME_SLICE, 5, 6)), axis=-1)
            i += 1
    print(dataset.shape)
    model = create_model(*dataset.shape[1:])
    model.summary()
