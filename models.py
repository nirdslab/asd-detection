from tensorflow import keras as k


def lstm_time_major(timesteps, ch_rows, ch_cols, bands):
    """
    Generate LSTM NN model, with first dimension as time
    """
    return k.models.Sequential(name='asd_lstm_32', layers=[
        # input
        k.layers.Input(shape=(timesteps, ch_rows, ch_cols, bands)),
        k.layers.TimeDistributed(k.layers.Flatten(), name='eeg'),
        # lstm 1
        k.layers.LSTM(32, return_sequences=True, kernel_regularizer='l1_l2', dropout=0.2, name='lstm_1'),
        # lstm 2
        k.layers.LSTM(64, kernel_regularizer='l1_l2', dropout=0.2, name='lstm_2'),
        # prediction
        k.layers.Dense(1, activation='sigmoid', kernel_regularizer='l1_l2', name='prediction')
    ])


def conv_net_time_major(timesteps, ch_rows, ch_cols, bands):
    """
    Generate 1D-convolution NN model, with first dimension as time
    """
    return k.models.Sequential(name='asd_conv_32', layers=[
        # input
        k.layers.Input(shape=(timesteps, ch_rows, ch_cols, bands)),
        k.layers.TimeDistributed(k.layers.Flatten(), name='eeg'),
        # convolution 1
        k.layers.Conv1D(filters=64, kernel_size=4, activation='relu', kernel_regularizer='l1_l2', padding='same', name='conv_1'),
        k.layers.MaxPooling1D(name='pool_1'),
        k.layers.Dropout(0.2, name='dropout_1'),
        k.layers.BatchNormalization(name='batch_norm_1'),
        # convolution 2
        k.layers.Conv1D(filters=128, kernel_size=4, activation='relu', kernel_regularizer='l1_l2', padding='same', name='conv_2'),
        k.layers.MaxPooling1D(name='pool_2'),
        k.layers.Dropout(0.2, name='dropout_2'),
        k.layers.BatchNormalization(name='batch_norm_2'),
        # prediction
        k.layers.Flatten(name='flatten'),
        k.layers.Dense(1, activation='sigmoid', kernel_regularizer='l1_l2', name='prediction')
    ])
