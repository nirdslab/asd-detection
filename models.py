from tensorflow import keras as k
from tensorflow.keras import layers as kl, models as km


def lstm_nn(timesteps, ch_rows, ch_cols, bands):
    """
    Generate LSTM NN model, with temporal dimension addressed first
    """
    return km.Sequential(name='asd_lstm', layers=[
        # input
        kl.Input(shape=(timesteps, ch_rows, ch_cols, bands)),
        kl.TimeDistributed(kl.Flatten(), name='eeg'),
        # lstm 1
        kl.LSTM(32, return_sequences=True, kernel_regularizer='l1_l2', dropout=0.2, name='lstm_1'),
        # lstm 2
        kl.LSTM(64, kernel_regularizer='l1_l2', dropout=0.2, name='lstm_2'),
        # prediction
        kl.Dense(1, activation='sigmoid', kernel_regularizer='l1_l2', name='prediction')
    ])


def conv_nn_time_major(timesteps, ch_rows, ch_cols, bands):
    """
    Generate 1D-convolution NN model, with temporal dimension addressed first
    """
    return km.Sequential(name='asd_conv_tm', layers=[
        # input
        kl.Input(shape=(timesteps, ch_rows, ch_cols, bands)),
        kl.TimeDistributed(kl.Flatten(), name='eeg'),
        # convolution 1
        kl.Conv1D(filters=64, kernel_size=4, activation='relu', kernel_regularizer='l1_l2', padding='same', name='conv_1'),
        kl.MaxPooling1D(name='pool_1'),
        kl.Dropout(0.2, name='dropout_1'),
        kl.BatchNormalization(name='b_norm_1'),
        # convolution 2
        kl.Conv1D(filters=128, kernel_size=4, activation='relu', kernel_regularizer='l1_l2', padding='same', name='conv_2'),
        kl.MaxPooling1D(name='pool_2'),
        kl.Dropout(0.2, name='dropout_2'),
        kl.BatchNormalization(name='b_norm_2'),
        # prediction
        kl.Flatten(name='flatten'),
        kl.Dense(1, activation='sigmoid', kernel_regularizer='l1_l2', name='prediction')
    ])


def conv_nn_channel_major(ch_rows, ch_cols, timesteps, bands):
    """
    Generate 1D-convolution NN model, with channel dimension addressed first
    """
    return km.Sequential(name='asd_conv_cm', layers=[
        # input
        kl.Input(shape=(ch_rows, ch_cols, timesteps, bands)),
        # reshape input to dimensions (channels, timesteps, bands)
        kl.Reshape(target_shape=(ch_rows * ch_cols, timesteps, bands), name='eeg'),
        # convolution 1
        kl.TimeDistributed(kl.Conv1D(filters=64, kernel_size=4, activation='relu', kernel_regularizer='l1_l2', padding='same'), name='conv_1'),
        kl.TimeDistributed(kl.MaxPooling1D(), name='pool_1'),
        kl.TimeDistributed(kl.Dropout(0.2), name='dropout_1'),
        kl.TimeDistributed(kl.BatchNormalization(), name='b_norm_1'),
        # convolution 2
        kl.TimeDistributed(kl.Conv1D(filters=128, kernel_size=4, activation='relu', kernel_regularizer='l1_l2', padding='same'), name='conv_2'),
        kl.TimeDistributed(kl.MaxPooling1D(), name='pool_2'),
        kl.TimeDistributed(kl.Dropout(0.2), name='dropout_2'),
        kl.TimeDistributed(kl.BatchNormalization(), name='b_norm_2'),
        # prediction
        k.layers.Flatten(name='flatten'),
        k.layers.Dense(1, activation='sigmoid', kernel_regularizer='l1_l2', name='prediction')
    ])
