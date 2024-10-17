# -*- coding: utf-8 -*-
"""
Author: huangxin
"""
import tensorflow as tf


def platform_app_filter_fn(record):
    """
    Filter function while reading tfrecord, this is an example of filter function.Pass filter function as a parameter to
    read_tfrecord function or ModelTrainer.The example is to filter out the record whose platform is "web".
    """
    example = tf.io.parse_single_example(record, {"platform": tf.io.FixedLenFeature(shape=(1,), dtype=tf.string)})
    return tf.squeeze(tf.math.not_equal(example["platform"], "web"))


def feature_fn_example(features, feature_description: dict):
    """
    Pass feature function as a parameter to read_tfrecord function or ModelTrainer.
    :param features: A dict mapping from feature name to tensor returned by tf.io.parse_single_example.
    :param feature_description:
    :return: A dict of treated features.
    """
    return features


def read_tfrecord(
        filenames,
        feature_description,
        batch_size, shuffle,
        shuffle_buffer_size=2048,
        num_parallel_calls=8,
        prefetch_num=1,
        labels=tuple(),
        use_weight=False,
        weight_name=None,
        filter_fn=None,
        feature_fn=None
):
    """
    Read tfrecord file and return a dataset object.
    :param filenames: A list of tfrecord file paths.
    :param feature_description: A dict mapping from feature name to tf.io.FixedLenFeature or tf.io.VarLenFeature.
    :param batch_size: The batch size of the dataset.
    :param shuffle: Whether to shuffle the dataset.
    :param shuffle_buffer_size: The buffer size of shuffle.
    :param num_parallel_calls: The number of parallel calls.
    :param prefetch_num: The number of prefetch.
    :param labels: A tuple of label names.
    :param use_weight: Whether to use weight for examples.
    :param weight_name: The name of the weight column.
    :param filter_fn: A filter function.Default is None, which means no filter.
    :param feature_fn: A feature treat function.Default is None, which means no treatment.
    :return: A dataset object which can be used to train model.
    """
    def _parse_example(example_proto):
        features = tf.io.parse_single_example(example_proto, feature_description)
        _labels = []
        for label in labels:
            _labels.append(features.pop(label))
        if feature_fn:
            features = feature_fn(features, feature_description)
        if use_weight:
            if not weight_name:
                raise AttributeError("The name of the weight column cannot be empty!")
            weight = features.pop(weight_name)
            return features, tuple(_labels), weight
        return features, tuple(_labels)

    def _input():
        dataset = tf.data.TFRecordDataset(filenames)
        if filter_fn:
            dataset = dataset.filter(filter_fn)
        dataset = dataset.map(_parse_example, num_parallel_calls=num_parallel_calls)
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        if prefetch_num > 0:
            dataset = dataset.prefetch(buffer_size=batch_size * prefetch_num)
        return dataset

    return _input()


def convert_to_tf_example(data, all_features: dict) -> tf.train.Example:
    """
    Convert data to tf.train.Example.
    :param data: dict, the key is the feature name, the value is the feature value.
    :param all_features: the dict of all features, the key is the feature name, the value is a tuple, the first element
    is the type of the feature, the second element is the length of the feature.
    :return: tf.train.Example
    """
    feature_dict = {}
    for key, value in all_features.items():
        if value[0] == STRING_TYPE:
            if value[1] == 1:
                feature_dict[key] = _bytes_feature([data[key].encode('utf-8')])
            else:
                feature_dict[key] = _bytes_feature([str(x).encode('utf-8') for x in data[key]])
        elif value[0] == INT_TYPE:
            if value[1] == 1:
                feature_dict[key] = _int64_feature([int(data[key])])
            else:
                feature_dict[key] = _int64_feature(list(map(int, data[key])))
        elif value[0] == FLOAT_TYPE:
            if value[1] == 1:
                feature_dict[key] = _float_feature([float(data[key])])
            else:
                feature_dict[key] = _float_feature(list(map(float, data[key])))
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def write_tf_record(data, path, all_features) -> None:
    """
    Write data to tfrecord file.
    :param data: A pandas DataFrame.
    :param path: path of the tfrecord file.
    :param all_features: the dict of all features, the key is the feature name, the value is a tuple, the first element
    is the type of the feature, the second element is the length of the feature.
    """
    with tf.io.TFRecordWriter(path) as writer:
        for index, row in data.iterrows():
            example = convert_to_tf_example(row, all_features)
            writer.write(example.SerializeToString())


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


STRING_TYPE = 'string'
INT_TYPE = 'int64'
FLOAT_TYPE = 'float32'
TYPE_DICT = {STRING_TYPE: tf.string, INT_TYPE: tf.int64, FLOAT_TYPE: tf.float32}
