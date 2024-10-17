# -*- coding: utf-8 -*-
"""
Author: huangxin
"""
import tensorflow as tf

from utils import tf_udf


if __name__ == '__main__':
    a = tf.constant([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0]], dtype=tf.float32)
    # print(tf_udf.get_mask(tf.constant([1], dtype=tf.int32), max_len=3, elem_type=tf.float32))
    b = tf_udf.get_mask(a, pad_element=0.0)
    print(b)
    c = tf_udf.reduce_mean_with_mask(tf.expand_dims(a, axis=-1), b)
    print(c)
