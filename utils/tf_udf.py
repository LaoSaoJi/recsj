# -*- coding: utf-8 -*-
"""
Author: huangxin
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing


def _embedding_res_check_reshape(embedding_res):
    """
    Check the shape of the result of keras.Embedding layer, if the shape is (None, 1, embedding_dim),
    then reshape it to (None, embedding_dim)
    :param embedding_res: Tensor
    :return: Tensor
    """
    if embedding_res.shape[1] == 1:
        return tf.reshape(embedding_res, shape=(-1, embedding_res.shape[-1]))
    else:
        return embedding_res


def embedding_return(func):
    def wrapper(**kwargs):
        inp = kwargs.pop("inputs")
        preprocess_layer, embedding_layer = func(**kwargs)
        if isinstance(inp, (tuple, list)):
            res = []
            for each in inp:
                res.append(_embedding_res_check_reshape(embedding_layer(preprocess_layer(each))))
            return res
        else:
            return _embedding_res_check_reshape(embedding_layer(preprocess_layer(inp)))

    return wrapper


@embedding_return
def hash_and_embedding_v2(treat_param, embedding_dimension, embedding_regularizer, embedding_initializer, name):
    """
    :param treat_param: the amount of hash buckets
    :param embedding_dimension: the dimension of output embedding vector
    :param embedding_regularizer: regularizer for embedding layer
    :param embedding_initializer: initializer for embedding layer
    :param name: the name of the embedding layer and hash layer.Generally speaking, use the feature name
    :return: Tensor，embedding vector
    """
    hash_layer = preprocessing.Hashing(num_bins=treat_param, name=f"{name}_hash_layer")
    embedding_layer = keras.layers.Embedding(input_dim=treat_param,
                                             output_dim=embedding_dimension,
                                             embeddings_regularizer=embedding_regularizer,
                                             embeddings_initializer=embedding_initializer,
                                             name=f"{name}_embedding_layer")
    return hash_layer, embedding_layer


@embedding_return
def string_lookup_embedding(treat_param, embedding_dimension, embedding_regularizer, embedding_initializer, name):
    """
    :param treat_param: the vocabulary of the feature, list or the path of the vocabulary file
    :param embedding_dimension: the dimension of output embedding vector
    :param embedding_regularizer: regularizer for embedding layer
    :param embedding_initializer: initializer for embedding layer
    :param name: the name of the embedding layer and hash layer.Generally speaking, use the feature name
    :return: Tensor，embedding vector
    """
    string_lookup_layer = preprocessing.StringLookup(vocabulary=treat_param,
                                                     name=f"{name}_lookup_layer")
    embedding_layer = keras.layers.Embedding(input_dim=len(string_lookup_layer.get_vocabulary()) + 2,
                                             output_dim=embedding_dimension,
                                             embeddings_regularizer=embedding_regularizer,
                                             embeddings_initializer=embedding_initializer,
                                             name=f"{name}_embedding_layer")

    return string_lookup_layer, embedding_layer


@embedding_return
def integer_lookup_embedding(treat_param, embedding_dimension, embedding_regularizer, embedding_initializer, name):
    """
    :param treat_param: the vocabulary of the feature, list or the path of the vocabulary file
    :param embedding_dimension: the dimension of output embedding vector
    :param embedding_regularizer: regularizer for embedding layer
    :param embedding_initializer: initializer for embedding layer
    :param name: the name of the embedding layer and hash layer.Generally speaking, use the feature name
    :return: Tensor，embedding vector
    """
    integer_lookup_layer = preprocessing.IntegerLookup(vocabulary=treat_param,
                                                       mask_value=None,
                                                       name=f"{name}_lookup_layer")
    embedding_layer = keras.layers.Embedding(input_dim=len(integer_lookup_layer.get_vocabulary()) + 2,
                                             output_dim=embedding_dimension,
                                             embeddings_regularizer=embedding_regularizer,
                                             embeddings_initializer=embedding_initializer,
                                             name=f"{name}_embedding_layer")

    return integer_lookup_layer, embedding_layer


def get_mask(seq, pad_element="<nan>", dtype=tf.float32):
    return tf.reshape(tf.cast(tf.not_equal(seq, pad_element), dtype=dtype), shape=(-1, seq.shape[1], 1))


def reduce_mean_with_mask(embeddings, mask, dtype=tf.float32):
    masked_embeddings = embeddings * mask
    summed = tf.reduce_sum(masked_embeddings, axis=1)
    valid_length = tf.reduce_sum(mask, axis=1)
    return summed / tf.cast(valid_length, dtype=dtype)
