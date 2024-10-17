# -*- coding: utf-8 -*-
"""
Author: huangxin
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class SecondOrderCross(keras.layers.Layer):
    """
    The second order cross layer in FM model
    """

    def __init__(self, **kwargs):
        super(SecondOrderCross, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SecondOrderCross, self).build(input_shape)

    def call(self, inputs, **kwargs):
        square_of_sum = tf.square(tf.reduce_sum(inputs, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(tf.square(inputs), axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2)
        return cross_term


class Dice(keras.layers.Layer):
    """
    Dice activation function
    """

    def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        self.batch_normalization = None
        self.alpha = None
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.batch_normalization = keras.layers.BatchNormalization(axis=self.axis, epsilon=self.epsilon, center=False,
                                                                   scale=False)
        self.alpha = self.add_weight(shape=(input_shape[-1],), initializer=keras.initializers.Zeros(),
                                     dtype=tf.float32, name="Dice_alpha")
        super(Dice, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        inputs_normed = self.batch_normalization(inputs, training=training)
        x_p = tf.sigmoid(inputs_normed)
        return self.alpha * (1.0 - x_p) * inputs + x_p * inputs


def get_activation(ac):
    activation = None
    if ac is None:
        pass
    if isinstance(ac, str):
        if ac == "Dice":
            activation = Dice()
        elif ac == "relu":
            activation = keras.activations.relu
        elif ac == "selu":
            activation = keras.activations.selu
        elif ac == "sigmoid":
            activation = keras.activations.sigmoid
        else:
            raise AttributeError("激活函数输入错误")
    elif isinstance(ac, type(keras.activations.softmax)) or isinstance(ac, type(tf.nn.sigmoid)):
        activation = ac
    return activation


class FC(keras.layers.Layer):
    """
    several successive dense layers
    """

    def __init__(self,
                 tower_units: list,
                 tower_name: str,
                 hidden_activation,
                 regularizer=keras.regularizers.L2(0.00001),
                 use_bn=True,
                 dropout=0.0,
                 output_activation=None,
                 seed=2023,
                 **kwargs):
        """
        :param tower_units: units of each layer
        :param tower_name: name of the tower
        :param hidden_activation: activation of hidden layers
        :param regularizer: regularizer
        :param use_bn: whether to use batch normalization
        :param dropout: whether to use dropout.If you want to use dropout, the value is the dropout rate;
               otherwise, the value is 0.0
        :param output_activation: activation of output layer
        :param seed: random seed
        """
        self.tower_units = tower_units
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.regularizer = regularizer
        self.use_bn = use_bn
        self.tower_name = tower_name
        self.seed = seed
        self.dropout = dropout

        self.kernels = None
        self.biases = None
        self.activations = None
        if self.use_bn:
            self.batch_normalizations = None
        self.dropout_layers = None

        super(FC, self).__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        kernel_units = [int(input_size)] + list(self.tower_units)
        self.kernels = [self.add_weight(name=f"{self.tower_name}_kernel_{i}",
                                        shape=(kernel_units[i], kernel_units[i + 1]),
                                        initializer=keras.initializers.glorot_normal(seed=self.seed),
                                        regularizer=self.regularizer,
                                        trainable=True) for i in range(len(self.tower_units))]
        self.biases = [self.add_weight(name=f"{self.tower_name}_bias_{i}",
                                       shape=(self.tower_units[i],),
                                       initializer=keras.initializers.Zeros(),
                                       trainable=True) for i in range(len(self.tower_units))]
        if self.use_bn:
            self.batch_normalizations = [keras.layers.BatchNormalization() for i in range(len(self.tower_units))]

        self.activations = [get_activation(self.hidden_activation) for _ in range(len(self.tower_units) - 1)] + \
                           [get_activation(self.output_activation)]

        if self.dropout > 0.0:
            self.dropout_layers = [keras.layers.Dropout(self.dropout) for i in range(len(self.tower_units))]

        super(FC, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        tower_input = inputs
        for i in range(len(self.tower_units)):
            cur = tf.nn.bias_add(tf.tensordot(tower_input, self.kernels[i], axes=(-1, 0)), self.biases[i])
            if self.use_bn:
                cur = self.batch_normalizations[i](cur, training=training)
            try:
                cur = self.activations[i](cur, training=training)
            except TypeError:
                if self.activations[i]:
                    cur = self.activations[i](cur)
            if self.dropout > 0.0:
                cur = self.dropout_layers[i](cur)
            tower_input = cur

        return tower_input


def din_attention_unit(tower_units, name):
    """
    din attention unit
    :param tower_units: dense layer units of attention unit
    :param name: name of the attention unit, used to distinguish different attention units
    """
    return FC(tower_units=tower_units,
              tower_name=name,
              hidden_activation="Dice",
              name=name + "_layer")


class TargetAttention(keras.layers.Layer):
    """
    -inputs: [query, keys]
        - query-shape: (batch_size, embedding_dim) or (batch_size, 1, embedding_dim)
        - keys-shape: (batch_size, seq_len, embedding_dim)
    - outputs: attention weights
        -shape: (batch_size, seq_len, 1)
    """

    def __init__(self, att_units, seq_len, embedding_dim, use_softmax=True, tower_name="", **kwargs):
        """
        :param att_units:
        :param seq_len:
        :param embedding_dim:
        :param use_softmax:
        :param tower_name:
        :param kwargs:
        """
        self.att_module = din_attention_unit(att_units, tower_name)
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.use_softmax = use_softmax
        super(TargetAttention, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        query_dim = len(K.int_shape(inputs[0]))
        if query_dim == 2:
            query = tf.reshape(inputs[0], shape=(-1, 1, self.embedding_dim))
        else:
            assert query_dim == 3, "incorrect query shape!"
            query = inputs[0]
        queries = tf.tile(query, multiples=[1, self.seq_len, 1])
        keys = inputs[1]
        att_unit_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        att_module_output = self.att_module(att_unit_input)
        if self.use_softmax:
            att_module_output = tf.squeeze(att_module_output, axis=-1)
            att_module_output = tf.nn.softmax(att_module_output)
            att_module_output = tf.expand_dims(att_module_output, axis=-1)
        return att_module_output  # (None, seq_len, 1)


class SimpleSelfAttention(keras.layers.Layer):
    """
    self attention layer
    input_shape:
        - (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(self, mask=None, use_scale=True, **kwargs):
        self.mask = mask
        self.use_scale = use_scale
        super(SimpleSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SimpleSelfAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if len(K.int_shape(inputs)) != 3:
            raise ValueError("incorrect input shape!")
        inp = inputs
        _product = tf.matmul(inp, inp, transpose_b=True)  # (batch_size, sequence_length, sequence_length)
        if self.use_scale:
            _product = _product / inp.shape[-1] ** 0.5
        normalized_att_score = tf.nn.softmax(_product, axis=-1)
        result = tf.matmul(normalized_att_score, inp)  # (batch_size, sequence_length, embedding_dimension)
        if self.mask is not None:
            result = result * tf.reshape(self.mask, shape=(-1, self.mask.shape[1], 1))
        return result


class LinearLayer(keras.layers.Layer):

    def __init__(self, task_name, regularizer, activation=None, **kwargs):
        self.task_name = task_name
        self.regularizer = regularizer
        self.activation = activation
        self.kernel = None
        self.bias = None
        super(LinearLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name=f"{self.task_name}_output_kernel",
                                      shape=(input_shape[-1], 1),
                                      initializer=keras.initializers.glorot_normal(seed=1024),
                                      regularizer=self.regularizer,
                                      trainable=True)
        self.bias = self.add_weight(name=f"{self.task_name}_output_bias",
                                    shape=(1,),
                                    initializer=keras.initializers.Zeros(),
                                    trainable=True)
        self.activation = get_activation(self.activation)
        super(LinearLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        _input = inputs
        res = tf.nn.bias_add(tf.tensordot(_input, self.kernel, axes=(-1, 0)), self.bias)
        if self.activation:
            res = self.activation(res)
        return res


class ExpertNetwork(keras.layers.Layer):
    """
    expert network,which is used in SEI module of HiNet and MMoE
    """

    def __init__(self,
                 expert_num,
                 expert_units,
                 gate_num,
                 gate_units,
                 dnn_reg,
                 dnn_act,
                 use_bn,
                 network_name,
                 dropout=0.0,
                 seed_root=2023,
                 gate_stop_gradient=False,
                 **kwargs):
        """
        :param expert_num: amount of experts
        :param expert_units: fc units of each expert
        :param gate_num: amount of gates, which is equal to the amount of tasks
        :param gate_units: fc units of each gate
        :param dnn_reg: the regularizer of dnn
        :param dnn_act: the activation of dnn
        :param use_bn: whether to use batch normalization in dnn
        :param network_name: name
        :param dropout: whether to use dropout in dnn
        :param seed_root: random seed
        :param gate_stop_gradient: whether to stop gradient of gate
        """
        assert gate_units[-1] == expert_num
        self.experts = []
        self.expert_num = expert_num
        self.expert_units = expert_units
        self.gates = []
        self.gate_num = gate_num
        self.gate_units = gate_units
        self.dnn_reg = dnn_reg
        self.dnn_act = dnn_act
        self.use_bn = use_bn
        self.network_name = network_name
        self.seed_root = seed_root
        self.dropout = dropout
        if self.dropout > 0:
            self.dropout_layers = None
        self.gate_stop_gradient = gate_stop_gradient
        super(ExpertNetwork, self).__init__(**kwargs)
        self._sub_layers = {}

    def build(self, input_shape):
        for i in range(self.expert_num):
            cur_expert = FC(tower_units=self.expert_units,
                            tower_name=f"{self.network_name}_expert_{i}",
                            hidden_activation=self.dnn_act,
                            regularizer=self.dnn_reg,
                            use_bn=self.use_bn,
                            dropout=self.dropout,
                            output_activation=self.dnn_act,
                            seed=self.seed_root + i,
                            name=f"{self.network_name}_expert_{i}")
            self.experts.append(cur_expert)
            self._sub_layers[f"expert_{i}"] = cur_expert

        for i in range(self.gate_num):
            cur_gate = FC(tower_units=self.gate_units,
                          tower_name=f"{self.network_name}_gate_{i}",
                          hidden_activation=self.dnn_act,
                          regularizer=self.dnn_reg,
                          use_bn=self.use_bn,
                          output_activation=keras.activations.softmax,
                          seed=self.seed_root + self.expert_num + i,
                          name=f"{self.network_name}_gate_{i}")
            self.gates.append(cur_gate)
            self._sub_layers[f"gate_{i}"] = cur_gate

        if self.dropout > 0:
            self.dropout_layers = [keras.layers.Dropout(rate=self.dropout,
                                                        name=f"{self.network_name}_dropout_{i}")
                                   for i in range(self.expert_num)]
        super(ExpertNetwork, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        expert_input = inputs
        gate_input = tf.stop_gradient(inputs) if self.gate_stop_gradient else inputs
        experts_out, gates_out = [], []
        for i in range(self.expert_num):
            cur = self.experts[i](expert_input)
            if self.dropout > 0.0:
                cur = self.dropout_layers[i](cur)
            experts_out.append(cur)

        for gate in self.gates:
            cur = tf.reshape(gate(gate_input), shape=(-1, self.expert_num, 1))
            gates_out.append(cur)

        res = []
        experts_out_matrix = tf.stack(experts_out, axis=1)
        for gate_out in gates_out:
            res.append(tf.reduce_sum(experts_out_matrix * gate_out, axis=1))

        # usually len(res) won't equal to 1, 1 is especially prepared for SEI module
        if len(res) == 1:
            return res[0]
        return res

    def get_layer(self, layer_name):
        return self._sub_layers[layer_name]


class PartitionedNormalization(keras.layers.Layer):
    """
    PN layer，used in STAR model.Do Normalization for each domain separately.
    """

    def __init__(self, num_domains, **kwargs):
        """
        :param num_domains: the amount of domains
        """
        self.num_domains = num_domains
        self.mini_batch_of_domain = [0 for _ in range(self.num_domains)]

        super(PartitionedNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.batch_norm_layers = [keras.layers.BatchNormalization() for _ in range(self.num_domains)]

        super(PartitionedNormalization, self).build(input_shape)

    def call(self, inputs, domain_indicator=None, **kwargs):
        """
        :param domain_indicator: the feature which can separate the data into different domains
        """
        pn_input = inputs
        domain_index = tf.cast(tf.expand_dims(tf.argmax(domain_indicator, axis=-1), axis=-1), dtype=tf.float32)
        idx = tf.cast(tf.expand_dims(tf.range(tf.shape(inputs)[0]), axis=1), dtype=tf.float32)
        # the purpose is to keep the order of the input data
        _complete_tensor = tf.concat([idx, pn_input, domain_index], axis=-1)
        for i in range(self.num_domains):
            self.mini_batch_of_domain[i] = tf.boolean_mask(
                _complete_tensor, tf.equal(_complete_tensor[:, -1], float(i)))
            cur_idx = tf.expand_dims(self.mini_batch_of_domain[i][:, 0], axis=1)
            cur_data = self.mini_batch_of_domain[i][:, 1:]
            normed_data = self.batch_norm_layers[i](cur_data)
            self.mini_batch_of_domain[i] = tf.concat([cur_idx, normed_data], axis=-1)

        unordered_res = tf.concat(self.mini_batch_of_domain, axis=0)[:, :-1]
        sorted_idx = tf.argsort(unordered_res[:, 0])
        return tf.gather(unordered_res, sorted_idx)[:, 1:]
