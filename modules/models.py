# -*- coding: utf-8 -*-
"""
Author: huangxin
"""
import tensorflow as tf
from tensorflow import keras

import layers


class FM(keras.layers.Layer):
    """
    Factorization Machine Model
    """
    def __init__(self, embedding_dim, regularizer, output_prob=True, seed=1028, **kwargs):
        """
        :param embedding_dim: embedding dimension
        :param regularizer: regularizer for embedding layer and dense layer
        :param output_prob: whether output probability, if True, a sigmoid layer will be added before output
        :param seed: random seed
        """
        self.regularizer = regularizer
        self.seed = seed
        self.embedding_dim = embedding_dim
        self.output_prob = output_prob
        self.flatten_dim = 0
        self.first_order = None
        self.bias = None
        self.second_order_cross = layers.SecondOrderCross()
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.flatten_dim = input_shape[-1]
        self.first_order = self.add_weight(name="FM_first_order",
                                           shape=(self.flatten_dim, 1),
                                           initializer=keras.initializers.glorot_normal(seed=self.seed),
                                           regularizer=self.regularizer,
                                           trainable=True)
        self.bias = self.add_weight(name="FM_bias",
                                    shape=(1,),
                                    initializer=keras.initializers.Zeros(),
                                    trainable=True)
        super(FM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs 的维度: (batch_size, field_size * embedding_size)
        if self.flatten_dim % self.embedding_dim != 0:
            raise AttributeError("The flatten_dim must be divisible by embedding_dim")
        else:
            field_size = self.flatten_dim // self.embedding_dim
        first_order_input = inputs
        second_order_input = tf.reshape(inputs, shape=(-1, field_size, self.embedding_dim))
        f = tf.matmul(first_order_input, self.first_order) + self.second_order_cross(second_order_input) + self.bias
        if self.output_prob:
            return tf.math.sigmoid(f)
        else:
            return f


class FMAlpha(keras.layers.Layer):
    """
    An adjusted version of Factorization Machine Model
    """

    def __init__(self, factor_order=4, kernel_regularizer=None, logit_output=True, **kwargs):
        super(FMAlpha, self).__init__(**kwargs)
        if kernel_regularizer is None:
            self.regularizer = keras.regularizers.L2(1e-4)
        else:
            self.regularizer = kernel_regularizer
        self.factor_order = factor_order
        self.logit_output = logit_output

    def build(self, input_shape):

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.w = self.add_weight(name="one",
                                 shape=(last_dim, 1),
                                 initializer="glorot_uniform",
                                 regularizer=self.regularizer,
                                 trainable=True)
        self.v = self.add_weight(name="two",
                                 shape=(last_dim, self.factor_order),
                                 initializer="glorot_uniform",
                                 regularizer=self.regularizer,
                                 trainable=True)
        self.b = self.add_weight(name="bias",
                                 shape=(1,),
                                 initializer="zeros",
                                 trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):

        xv = 0.5 * tf.reduce_sum(tf.math.square(tf.matmul(inputs, self.v)) - tf.matmul(inputs, tf.math.square(self.v)),
                                 axis=1)

        xv = tf.expand_dims(xv, -1)
        xw = tf.matmul(inputs, self.w)
        f = xw + self.b + xv
        if self.logit_output:
            output = tf.math.sigmoid(f)
        else:
            output = f

        return output


class STARLayer(keras.layers.Layer):
    """
    STAR model
    """

    def __init__(self, hidden_units, num_domains, activation="relu", output_activation=None, l2_reg=0,
                 dropout_rate=0.0, seed=10240, **kwargs):
        """
        :param hidden_units: fc layer units
        :param num_domains: amount of domains
        :param activation: fc layer activation
        :param output_activation: output layer activation
        :param l2_reg: l2 regularization
        :param dropout_rate: whether use dropout
        :param seed: random seed
        """
        self.hidden_units = hidden_units
        self.num_domains = num_domains
        self.activation = activation
        self.output_activation = output_activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.seed = seed

        super(STARLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        hidden_units = [input_shape[-1]] + list(self.hidden_units)
        self.shared_kernels = [self.add_weight(name=f"shared_kernel_{i}",
                                               shape=(hidden_units[i], hidden_units[i + 1]),
                                               initializer=keras.initializers.glorot_normal(seed=self.seed),
                                               regularizer=keras.regularizers.l2(self.l2_reg),
                                               trainable=True) for i in range(len(self.hidden_units))]
        self.shared_biases = [self.add_weight(name=f"shared_bias_{i}",
                                              shape=(hidden_units[i + 1],),
                                              initializer=keras.initializers.Zeros(),
                                              trainable=True) for i in range(len(self.hidden_units))]

        self.domain_kernels = [[self.add_weight(name=f"domain_{i}_kernel_{j}",
                                                shape=(hidden_units[j], hidden_units[j + 1]),
                                                initializer=keras.initializers.glorot_normal(seed=self.seed + i * j),
                                                regularizer=keras.regularizers.l2(self.l2_reg),
                                                trainable=True) for j in range(len(self.hidden_units))]
                               for i in range(self.num_domains)]
        self.domain_biases = [[self.add_weight(name=f"domain_{i}_bias_{j}",
                                               shape=(hidden_units[j + 1],),
                                               initializer=keras.initializers.Zeros(),
                                               trainable=True) for j in range(len(self.hidden_units))]
                              for i in range(self.num_domains)]

        self.activation_layers = [layers.get_activation(self.activation) for _ in range(len(self.hidden_units))]
        if self.output_activation:
            self.activation_layers[-1] = layers.get_activation(self.output_activation)

        super(STARLayer, self).build(input_shape)

    def call(self, inputs, indicator=None, **kwargs):
        """
        :params:
            -> domain_indicator: one-hot vector,shape=(batch_size,num_domains)
        """
        layer_inputs = inputs
        output_list = [layer_inputs for _ in range(self.num_domains)]
        for i in range(len(self.hidden_units)):
            for j in range(self.num_domains):
                output_list[j] = tf.nn.bias_add(tf.tensordot(
                    output_list[j], self.shared_kernels[i] * self.domain_kernels[j][i], axes=(-1, 0)),
                    self.shared_biases[i] + self.domain_biases[j][i])
                output_list[j] = self.activation_layers[i](output_list[j])

        domain_indicator = tf.cast(indicator, dtype=tf.float32)
        output = tf.reduce_sum(tf.stack(output_list, axis=1) * tf.expand_dims(domain_indicator, axis=-1), axis=1)
        return output

    def get_config(self):
        config = {
            "hidden_units": self.hidden_units,
            "num_domains": self.num_domains,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "l2_reg": self.l2_reg,
            "dropout_rate": self.dropout_rate,
            "seed": self.seed
        }
        base_config = super(STARLayer, self).get_config()
        config.update(base_config)
        return config
