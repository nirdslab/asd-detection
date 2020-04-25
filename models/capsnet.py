#!/usr/bin/env python3
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras as k
from tensorflow.keras.datasets import mnist

# Set random seeds so that the same outputs are generated always
np.random.seed(42)
tf.random.set_seed(42)


class Activations:

    @staticmethod
    def squash(_data, axis=-1):
        """
        Normalize to unit vectors
        :param _data: Tensor with rank >= 2
        :param axis:
        :return:
        """
        square_sum = tf.reduce_sum(tf.square(_data), axis=axis, keepdims=True)
        squash_factor = square_sum / (1. + square_sum)
        unit_vector = _data / tf.sqrt(square_sum + k.backend.epsilon())
        return squash_factor * unit_vector


class Losses:

    @staticmethod
    def margin_loss(_y_true, _y_pred, _m_p=0.9, _m_n=0.1, _lambda=0.5):
        """
        Loss Function
        :param _y_true: shape: (None, num_caps)
        :param _y_pred: shape: (None, num_caps)
        :param _m_p: threshold for positive
        :param _m_n: threshold for negative
        :param _lambda: loss weight for negative
        :return: margin loss. shape: (None, )
        """
        p_err = tf.maximum(0., _m_p - _y_pred)  # shape: (None, num_caps)
        n_err = tf.maximum(0., _y_pred - _m_n)  # shape: (None, num_caps)
        p_loss = _y_true * tf.square(p_err)  # shape: (None, num_caps)
        n_loss = (1.0 - _y_true) * tf.square(n_err)  # shape: (None, num_caps)
        loss = tf.reduce_mean(p_loss + _lambda * n_loss, axis=-1)  # shape: (None, )
        return loss

    @staticmethod
    def reconstruction_loss(_y_true, _y_pred):
        """
        Mean Squared Error

        :param _y_true: shape: (None, 28, 28, 1)
        :param _y_pred: shape: (None, 28, 28, 1)
        :return:
        """
        return tf.reduce_mean(tf.square(_y_true - _y_pred))


class Metrics:

    @staticmethod
    def accuracy(_y_true, _y_pred):
        """
        :param _y_true: shape: (None, num_caps)
        :param _y_pred: shape: (None, num_caps)
        :return:
        """
        _y_pred = tf.argmax(_y_pred, axis=-1)
        _y_true = tf.argmax(_y_true, axis=-1)
        correct = tf.equal(_y_true, _y_pred)
        return tf.reduce_mean(tf.cast(correct, tf.float32))


class CapsConv2D(k.layers.Conv2D):

    def __init__(self, caps_layers, caps_dims, kernel_size, **kwargs):
        self.caps_layers = caps_layers
        self.caps_dims = caps_dims
        super().__init__(self.caps_layers * self.caps_dims, kernel_size, **kwargs)

    def call(self, inputs, **kwargs):
        result = super(CapsConv2D, self).call(inputs)
        result = tf.reshape(result, shape=(-1, result.shape[1], result.shape[2], result.shape[3] // self.caps_dims, self.caps_dims))
        return Activations.squash(result, axis=-1)


class CapsDense(k.layers.Layer):

    def __init__(self, num_caps, dim_caps, routing_iter, trainable=True, name=None, dtype=None, dynamic=False,
                 **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.routing_iter = routing_iter
        self.p_num_caps = ...
        self.p_dim_caps = ...
        self.w = ...

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_caps': self.num_caps,
            'dim_caps': self.dim_caps,
            'routing_iter': self.routing_iter,
            'p_num_caps': self.p_num_caps,
            'p_dim_caps': self.p_dim_caps
        })
        return config

    def build(self, input_shape: tf.TensorShape):
        assert input_shape.rank == 5
        rows, cols, cap_layers = input_shape[1], input_shape[2], input_shape[3]
        self.p_num_caps = rows * cols * cap_layers
        self.p_dim_caps = input_shape[4]
        self.w = self.add_weight(
            shape=(1, self.p_num_caps, self.num_caps, self.dim_caps, self.p_dim_caps),
            dtype=tf.float32,
            initializer='random_normal'
        )
        self.built = True

    @staticmethod
    def apply_routing_weights(_weights, _prediction):
        """
        Weight the prediction by routing weights, squash it, and return it
        :param _weights: (batch_size, p_num_caps, num_caps, 1, 1)
        :param _prediction: (batch_size, p_num_caps, num_caps, dim_caps, 1)
        :return:
        """
        # softmax of weights over num_caps axis
        softmax_routing = tf.nn.softmax(_weights, axis=2)
        '''shape: (batch_size, p_num_caps, num_caps, 1, 1)'''

        # elementwise multiplication of weights with prediction
        w_prediction = tf.multiply(softmax_routing, _prediction)
        '''shape: (batch_size, p_num_caps, num_caps, dim_caps, 1)'''

        # sum over p_num_caps axis
        w_prediction_sum = tf.reduce_sum(w_prediction, axis=1, keepdims=True)
        '''shape: (batch_size, 1, num_caps, dim_caps, 1)'''

        squashed_w_prediction_sum = Activations.squash(w_prediction_sum, axis=-2)
        '''shape: (batch_size, 1, num_caps, dim_caps, 1)'''

        return squashed_w_prediction_sum

    def call(self, inputs, **kwargs):
        # get batch size of input
        batch_size = tf.shape(inputs)[0]
        # reshape input
        flattened = tf.reshape(inputs, (batch_size, self.p_num_caps, self.p_dim_caps))
        batch_input = tf.expand_dims(flattened, axis=-1)
        '''shape: (batch_size, p_num_caps, p_dim_caps, 1)'''
        batch_input = tf.expand_dims(batch_input, axis=2)
        '''shape: (batch_size, p_num_caps, 1, p_dim_caps, 1)'''
        batch_input = tf.tile(batch_input, [1, 1, self.num_caps, 1, 1])
        '''shape: (batch_size, p_num_caps, num_caps, p_dim_caps, 1)'''

        # tile transformation matrix for each element in batch
        batch_w = tf.tile(self.w, [batch_size, 1, 1, 1, 1])
        '''shape: (batch_size, p_num_caps, num_caps, dim_caps, p_dim_caps)'''

        # calculate prediction (dot product of batch_w and batch_input)
        # this returns the matrix multiplication of last two dims, preserving previous dims
        prediction = tf.matmul(batch_w, batch_input)
        '''shape: (batch_size, p_num_caps, num_caps, dim_caps, 1)'''

        # ROUTING SECTION ----------
        # initialize routing weights to zero
        routing_weights = tf.zeros(shape=(batch_size, self.p_num_caps, self.num_caps, 1, 1), dtype=tf.float32)
        '''shape: (batch_size, p_num_caps, num_caps, 1, 1)'''

        @tf.function
        def dynamic_routing(w_routing):
            # update routing weights for routing_iter iterations
            for i in range(self.routing_iter):
                # step 1: getting weighted prediction
                w_prediction = self.apply_routing_weights(w_routing, prediction)
                '''shape: (batch_size, 1, num_caps, dim_caps, 1)'''
                # step 2: tile the weighted prediction for each previous capsule
                w_prediction_tiled = tf.tile(w_prediction, [1, self.p_num_caps, 1, 1, 1])
                '''shape: (batch_size, p_num_caps, num_caps, dim_caps, 1)'''
                # step 3: find the agreement between prediction and weighted prediction
                agreement = tf.matmul(prediction, w_prediction_tiled, transpose_a=True)
                '''shape: (batch_size, p_num_caps, num_caps, 1, 1)'''
                # update routing weights based on agreement
                w_routing = tf.add(w_routing, agreement)
            # return the final prediction after routing
            w_prediction = self.apply_routing_weights(w_routing, prediction)
            '''shape: (batch_size, 1, num_caps, dim_caps, 1)'''
            return w_prediction

        final_prediction = dynamic_routing(routing_weights)

        # reshape to (None, num_caps, dim_caps)
        return tf.reshape(final_prediction, shape=(-1, self.num_caps, self.dim_caps))


def safe_l2_norm(_data, axis=-1, keepdims=False):
    squared_norm = tf.reduce_sum(tf.square(_data), axis=axis, keepdims=keepdims)
    return tf.sqrt(squared_norm + k.backend.epsilon())


def max_mask(_data):
    """
    Mask data from all capsules except the most activated one, for each instance
    :param _data: shape: (None, num_caps, dim_caps)
    :return:
    """
    _norm = safe_l2_norm(_data, axis=-1)  # shape: (None, num_caps)
    _y_pred = tf.argmax(_norm, axis=-1)  # shape: (None, )
    _mask = tf.expand_dims(tf.one_hot(_y_pred, depth=_norm.shape[-1]), axis=-1)  # shape: (None, num_caps, 1)
    _masked = tf.multiply(_data, _mask)  # shape: (None, num_caps, dim_caps)
    return _masked


def create_capsnet_model(input_shape, name) -> k.Model:
    # input
    l1 = k.layers.Input(shape=input_shape, name='input')  # type: tf.Tensor
    # initial convolution
    l2 = k.layers.Conv2D(filters=256, kernel_size=(9, 9), strides=(1, 1), activation='relu', name='conv')(l1)  # type: tf.Tensor
    # convolution capsule layer
    l3 = CapsConv2D(caps_layers=32, caps_dims=8, kernel_size=(9, 9), strides=(2, 2), activation='relu', name='conv_caps')(l2)  # type: tf.Tensor
    # dense capsule layer with dynamic routing
    l4 = CapsDense(num_caps=10, dim_caps=16, routing_iter=3, name='dense_caps')(l3)  # type: tf.Tensor
    # decoder
    d0 = k.layers.Lambda(max_mask)(l4)  # type: tf.Tensor
    d1 = k.layers.Flatten()(d0)  # type: k.layers.Layer
    d2 = k.layers.Dense(512, activation='relu')(d1)  # type: tf.Tensor
    d3 = k.layers.Dense(1024, activation='relu')(d2)  # type: tf.Tensor
    d4 = k.layers.Dense(tf.reduce_prod(input_shape), activation='sigmoid')(d3)  # type: tf.Tensor
    # output layers
    margin = k.layers.Lambda(safe_l2_norm, name='margin')(l4)  # type: tf.Tensor
    reconstruction = k.layers.Reshape(input_shape, name='reconstruction')(d4)  # type: tf.Tensor
    # define the model
    return k.models.Model(inputs=l1, outputs=[margin, reconstruction], name=name)


if __name__ == '__main__':
    NUM_CLASSES = 10
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # one hot encode data
    x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]
    y_train, y_test = k.utils.to_categorical(y_train, NUM_CLASSES), k.utils.to_categorical(y_test, NUM_CLASSES)

    model = create_capsnet_model(input_shape=x_train.shape[1:], name='mnist_capsnet')
    model.compile(optimizer='adam', loss=[Losses.margin_loss, Losses.reconstruction_loss], loss_weights=[1e0, 5e-3],
                  metrics={'margin': Metrics.accuracy})
    model.summary()

    # checkpoint function to save best weights
    checkpoint = k.callbacks.ModelCheckpoint("best_weights.hdf5", save_best_only=True)

    if os.path.exists('best_weights.hdf5'):
        # load existing weights
        model.load_weights('best_weights.hdf5')
        # evaluation
        model.evaluate(x_test, [y_test, x_test])
    else:
        # training
        model.fit(x_train, [y_train, x_train], batch_size=50, epochs=5, validation_split=0.1, callbacks=[checkpoint])


    def print_results():
        indices = np.random.randint(0, len(x_test), 10)
        _n, _x, _y = len(indices), x_test[indices], y_test[indices]
        [_y_p, _x_p] = model.predict(_x)
        fig, axs = plt.subplots(ncols=5, nrows=4)
        for z in range(_n):
            i = (z // 5) * 2
            j = z % 5
            axs[i, j].imshow(np.squeeze(_x_p[z]), cmap='gray', vmin=0.0, vmax=1.0)
            axs[i, j].axis('off')
            axs[i + 1, j].imshow(np.squeeze(_x[z]), cmap='gray', vmin=0.0, vmax=1.0)
            axs[i + 1, j].axis('off')
        fig.show()


    print_results()
