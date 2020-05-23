#!/usr/bin/env python3

from capsnet.layers import ConvCaps2D, DenseCaps
from capsnet.nn import squash, norm, mask_cid
from tensorflow.keras import layers as kl, models as km

DROPOUT = 0.4
REG = 'l1_l2'
CONV_SPEC = {'padding': 'same', 'kernel_regularizer': REG}


class UnitBlock(kl.Layer):
    def __init__(self, filters, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bn = kl.BatchNormalization()
        self.relu = kl.ReLU()
        self.conv = kl.Conv2D(filters, kernel_size, **CONV_SPEC)
        self.drop = kl.Dropout(DROPOUT)

    def call(self, inputs, **kwargs):
        return self.drop(self.conv(self.relu(self.bn(inputs))))


class ConvBlock(kl.Layer):
    def __init__(self, filters, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block0 = UnitBlock(filters, kernel_size)
        self.block1 = UnitBlock(filters, kernel_size)
        self.concat = kl.Concatenate()

    def call(self, inputs, **kwargs):
        return self.concat([inputs, self.block1(self.block0(inputs))])


class TransitionBlock(kl.Layer):
    def __init__(self, filters, kernel_size=(1, 1), strides=(2, 1), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = UnitBlock(filters, kernel_size)
        self.pool = kl.AveragePooling2D(strides)

    def call(self, inputs, **kwargs):
        return self.pool(self.block(inputs))


class DenseBlock(kl.Layer):
    def __init__(self, conv, filters, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = conv
        self.layers = []
        for i in range(conv):
            self.layers.append(ConvBlock(filters, kernel_size))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


def CONV(timesteps, ch_rows, ch_cols, bands):
    """
    Generate Convolution Model for EEG data

    :param timesteps:
    :param ch_rows:
    :param ch_cols:
    :param bands:
    :return:
    """
    # == input layer(s) ==
    il = kl.Input(shape=(timesteps, ch_rows, ch_cols, bands))
    ml = kl.Reshape((timesteps, ch_rows * ch_cols, bands))(il)

    # == define variables ==
    _f = 8  # filters per convolution
    _l = 4  # convolutions per block
    _n = 4  # dense + transition blocks
    _k = (4, 1)  # size of convolution kernel

    # == intermediate layer(s) ==
    # initial convolution
    ml = kl.Conv2D(filters=_f, kernel_size=_k)(ml)
    ml = kl.BatchNormalization()(ml)
    ml = kl.ReLU()(ml)
    # stack dense blocks and transition blocks
    for i in range(_n):
        ml = DenseBlock(conv=_l, filters=_f, kernel_size=_k)(ml)
        ml = TransitionBlock(filters=ml.shape[-1])(ml)
    # global average pooling
    ml = kl.Flatten()(ml)

    # == output layer(s) ==
    label = kl.Dense(2, activation='softmax', kernel_regularizer=REG, name='l')(ml)
    score = kl.Dense(1, kernel_regularizer=REG, name='s')(ml)

    # == create and return model ==
    return km.Model(inputs=il, outputs=[label, score], name='CONV')


def LSTM(timesteps, ch_rows, ch_cols, bands):
    """
    Generate LSTM Model for EEG data

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
    N = 4
    seq = []
    # stack LSTM blocks
    for i in range(N):
        ml = kl.LSTM(B, return_sequences=True, dropout=DROPOUT)(ml)
        seq.append(ml)
        if i > 0: ml = kl.Concatenate()([*seq])
    # convolution-lstm layer 3
    ml = kl.LSTM(B, dropout=DROPOUT)(ml)
    ml = kl.Dense(B, activation='relu')(ml)

    # == output layer(s) ==
    label = kl.Dense(2, activation='softmax', kernel_regularizer=REG, name='l')(ml)
    score = kl.Dense(1, kernel_regularizer=REG, name='s')(ml)

    # == create and return model ==
    return km.Model(inputs=il, outputs=[label, score], name='LSTM')


def CAPS(timesteps, ch_rows, ch_cols, bands):
    """
    Generate Capsule Model for EEG data

    :param timesteps:
    :param ch_rows:
    :param ch_cols:
    :param bands:
    :return:
    """
    # == input layer(s) ==
    il = kl.Input(shape=(timesteps, ch_rows, ch_cols, bands))
    ml = kl.Reshape(target_shape=(timesteps, ch_rows * ch_cols, bands))(il)

    # == define variables ==
    _f = 8  # filters per convolution
    _l = 4  # convolutions per block
    _n = 4  # dense + transition blocks
    _d0 = 4  # start capsule dimensions
    _s = (2, 1)  # conv capsule stride
    _d1 = 8  # final capsule dimensions
    _r = 3  # dynamic routing iterations
    _k = (4, 1)  # size of convolution kernel

    # == intermediate layer(s) ==
    # initial convolution
    ml = kl.Conv2D(filters=_f, kernel_size=_k)(ml)
    ml = kl.BatchNormalization()(ml)
    ml = kl.ReLU()(ml)
    # stack dense blocks and transition blocks
    for i in range(_n):
        ml = DenseBlock(conv=_l, filters=_f, kernel_size=_k)(ml)
        ml = TransitionBlock(filters=ml.shape[-1])(ml)
    # convert to capsule domain
    ml = ConvCaps2D(filters=_f, filter_dims=_d0, kernel_size=_k, strides=_s)(ml)
    ml = kl.Lambda(squash)(ml)
    # dense capsule layer with dynamic routing
    ml = DenseCaps(caps=2, caps_dims=_d1, routing_iter=_r)(ml)
    ml = kl.Lambda(squash)(ml)
    # select capsule with highest activity
    cl = kl.Lambda(mask_cid)(ml)

    # == output layer(s) ==
    label = kl.Lambda(norm, name='l')(ml)
    score = kl.Dense(1, name='s')(cl)

    # == create and return model ==
    return km.Model(inputs=il, outputs=[label, score], name='CAPS')


def MLP(features):
    """
    Generate MLP for Thermal Data

    :param features:
    """
    # == input layer(s) ==
    il = kl.Input(shape=(features,))
    ml = il

    # == intermediate layer(s) ==
    ml = kl.Dense(16)(ml)
    ml = kl.Dense(32)(ml)
    ml = kl.Dense(64)(ml)

    # == output layer(s) ==
    label = kl.Dense(2, activation='softmax', kernel_regularizer=REG, name='l')(ml)
    score = kl.Dense(1, kernel_regularizer=REG, name='s')(ml)

    # == create and return model ==
    return km.Model(inputs=il, outputs=[label, score], name='CONV')
