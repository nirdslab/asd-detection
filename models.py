#!/usr/bin/env python3

from capsnet.layers import ConvCaps2D, DenseCaps
from capsnet.nn import squash, norm, mask_cid
from tensorflow.keras import layers as kl, models as km

DROPOUT = 0.4
REG = 'l1_l2'


def conv_nn(timesteps, ch_rows, ch_cols, bands):
    """
    Generate Convolution NN based model

    :param timesteps:
    :param ch_rows:
    :param ch_cols:
    :param bands:
    :return:
    """

    def DB(name, layers, filters, kernel_size):
        conv_spec = {'padding': 'same', 'kernel_regularizer': REG}

        def call(_ml):
            _layers = []
            for _i in range(layers):
                # convolution
                _ml = kl.BatchNormalization(name=f'db_bnrm_{name}_{_i + 1}')(_ml)
                _ml = kl.ReLU(name=f'db_relu_{name}_{_i + 1}')(_ml)
                _ml = kl.Conv2D(filters, kernel_size, **conv_spec, name=f'db_conv_{name}_{_i + 1}')(_ml)
                _ml = kl.Dropout(DROPOUT, name=f'db_drop_{name}_{_i + 1}')(_ml)
                _layers.append(_ml)
                if _i > 0: _ml = kl.Concatenate(name=f'c_{name}_{_i + 1}')([*_layers])
            return _ml

        return call

    def TB(name, filters):
        conv_spec = {'padding': 'same', 'kernel_regularizer': REG}

        def call(_ml):
            # convolution
            _ml = kl.BatchNormalization(name=f'tb_bnrm_{name}')(_ml)
            _ml = kl.ReLU(name=f'tb_relu_{name}')(_ml)
            _ml = kl.Conv2D(filters, (1, 1), **conv_spec, name=f'tb_conv_{name}')(_ml)
            _ml = kl.Dropout(DROPOUT, name=f'tb_dropout_{name}')(_ml)
            _ml = kl.AveragePooling2D((2, 1), name=f'tb_pool_{name}')(_ml)
            return _ml

        return call

    # == input layer(s) ==
    il = kl.Input(shape=(timesteps, ch_rows, ch_cols, bands))
    ml = kl.Reshape((timesteps, ch_rows * ch_cols, bands))(il)

    # == intermediate layer(s) ==
    # initial convolution
    _f = 8
    _k = (5, 1)
    ml = kl.Conv2D(filters=_f, kernel_size=_k, name=f'conv_1')(ml)
    # dense blocks
    for i in range(2):
        _l = 4 * (i + 1)
        ml = DB(name=i + 1, layers=_l, filters=_f, kernel_size=_k)(ml)
        ml = TB(name=i + 1, filters=_f * _l)(ml)
    # flatten
    ml = kl.Flatten(name='flatten_ol')(ml)

    # == output layer(s) ==
    ol_c = kl.Dense(2, activation='softmax', kernel_regularizer=REG, name='l')(ml)
    ol_r = kl.Dense(1, kernel_regularizer=REG, name='s')(ml)

    # == create and return model ==
    return km.Model(inputs=il, outputs=[ol_c, ol_r], name='asd_conv')


def lstm_nn(timesteps, ch_rows, ch_cols, bands):
    """
    Generate LSTM NN based model

    :param timesteps:
    :param ch_rows:
    :param ch_cols:
    :param bands:
    :return:
    """
    # == input layer(s) ==
    il = kl.Input(shape=(timesteps, ch_rows, ch_cols, bands))
    ml = kl.Reshape((timesteps, ch_rows * ch_cols * bands))(il)
    # == intermediate layer(s) ==
    B = 32
    N = 8
    seq = []
    for i in range(N):
        # convolution lstm
        ml = kl.LSTM(B, return_sequences=True, dropout=DROPOUT, name=f'h_lstm{i + 1}')(ml)
        seq.append(ml)
        if i > 0: ml = kl.Concatenate(name=f'h_concat_{i}')([*seq])
    # convolution-lstm layer 3
    ml = kl.LSTM(B, dropout=DROPOUT, name=f'f_lstm')(ml)
    # dense layer
    ml = kl.Dense(B, activation='relu')(ml)

    # == output layer(s) ==
    ol_c = kl.Dense(2, activation='softmax', kernel_regularizer=REG, name='l')(ml)
    ol_r = kl.Dense(1, kernel_regularizer=REG, name='s')(ml)

    # == create and return model ==
    return km.Model(inputs=il, outputs=[ol_c, ol_r], name='asd_lstm')


def capsule_nn(timesteps, ch_rows, ch_cols, bands):
    """
    Generate Capsule NN based model

    :param timesteps:
    :param ch_rows:
    :param ch_cols:
    :param bands:
    :return:
    """
    # == input layer(s) ==
    il = kl.Input(shape=(timesteps, ch_rows, ch_cols, bands))

    # == intermediate layer(s) ==
    ml = kl.Reshape(target_shape=(timesteps, ch_rows * ch_cols, bands), name='eeg')(il)
    # initial convolution
    ml = kl.Conv2D(filters=64, kernel_size=(5, 1), strides=(1, 1), activation='relu', name='conv')(ml)
    # convert to capsule domain
    ml = ConvCaps2D(filters=16, filter_dims=8, kernel_size=(5, 1), strides=(2, 1), name='conv_caps')(ml)
    ml = kl.Lambda(squash)(ml)
    # dense capsule layer with dynamic routing
    ml = DenseCaps(caps=2, caps_dims=8, routing_iter=3, name='dense_caps')(ml)
    ml = kl.Lambda(squash)(ml)
    # select capsule with highest activity
    cl = kl.Lambda(mask_cid)(ml)

    # == output layer(s) ==
    label = kl.Lambda(norm, name='l')(ml)
    score = kl.Dense(1, name='s')(cl)

    # == create and return model ==
    return km.Model(inputs=il, outputs=[label, score], name='asd_caps')
