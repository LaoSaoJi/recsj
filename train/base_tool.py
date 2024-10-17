# -*- coding: utf-8 -*-
"""
Author: huangxin
"""
import copy
import importlib
import json
import multiprocessing
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import total_ordering
from typing import List, Generator, Iterable

import tensorflow as tf
from tensorflow import feature_column as tffc
from tensorflow import keras

# from utils import oss_util
# from utils.data_util import execute_sql
from utils.tf_record_util import TYPE_DICT, read_tfrecord, convert_to_tf_example

json_base_path = "model_features"
crontab_mode = "cron"
fill_mode = "fill"
writer_mode_write = "write"
writer_mode_debug = "debug"


class MultiIODict(dict):
    """
    A dict that you can get or set values of several keys at one time.
    Example:
        a = MultiIODict()
        a["a", "b"] = 1, 2
        print(a["a", "b"])
    """

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            return super().__getitem__(item)
        res = []
        for key in item:
            res.append(super().__getitem__(key))
        return res

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            super().__setitem__(key, value)
            return
        assert len(key) == len(value), "key 和 value 数量不一致"
        for each in zip(key, value):
            super().__setitem__(*each)


@dataclass
class DataDownloadContext:
    data_path: str
    start_days: int
    end_days: int
    mode: str
    mlp: bool
    workers: int


class TFRecordBaseWriter(object):

    def __init__(self
                 , base_sql: str
                 , ctx: DataDownloadContext
                 , data_version: str
                 , threshold: int = 0):
        self.base_sql = base_sql
        self.all_features = json.load(open(f"{json_base_path}/{data_version}.json"))
        self.ctx = ctx
        self.start_date = datetime.now() - timedelta(ctx.start_days + ctx.end_days)
        self.threshold = threshold
        self.mode = writer_mode_write
        self.debug_features = None

    def debug_mode(self, debug_features: list = None):
        """
        In debug mode, the tfrecord file is not actually written,
        but rather the results returned by the treat_ eat_row method are printed out
        """
        print("enter debug mode")
        self.mode = writer_mode_debug
        print(f"features you want to debug:{debug_features}")
        self.debug_features = debug_features

    def write_one_file(self, t: datetime) -> None:
        raise NotImplementedError

    def fill_day(self, date: datetime) -> None:
        raise NotImplementedError

    def delete_obsolete(self, del_days: int):
        raise NotImplementedError

    @staticmethod
    def treat_one_row(row: Iterable) -> dict:
        """
        If your data is from a generator or from an iterable object, you can rewrite this method
        to process each row of data
        row: a row of your data
        output: A dict;Its key refers to feature name, and its value refers to feature value
        If there is a problem with a column of data in a certain row, simply ending the function
        and returning an empty dictionary will be counted as a row of dirty data
        """
        pass

    def crontab(self) -> None:
        now = datetime.now()
        now_minus1_str = ""
        print(f"当前时间: {now.month}月{now.day}日{now.hour}时")
        mro_names = [x.__name__ for x in type(self).__mro__]
        for class_name in mro_names:
            if "Hourly" in class_name:
                now_minus1_str = (now - timedelta(hours=1)).strftime('%Y%m%d%H')
                break
            elif "Daily" in class_name:
                now_minus1_str = (now - timedelta(days=1)).strftime('%Y%m%d')
                break
        print(f"本次获取pt={now_minus1_str}")
        self.write_one_file(now)

    def fill_days(self, start, end):
        for i in range(start, end + 1):
            self.fill_day(self.start_date + timedelta(days=i))

    @staticmethod
    def data_generator(sql: str) -> Generator:
        # Example: each record is a row of data
        # with execute_sql(sql).open_reader() as reader:
        #     for record in reader:
        #         yield record
        pass

    def fill(self) -> None:
        ctx = self.ctx
        if ctx.mlp:
            pool = multiprocessing.Pool(processes=ctx.workers)
            step = ctx.start_days // ctx.workers
            date_range = [(x, x + step - 1) for x in range(ctx.end_days, ctx.start_days + ctx.end_days, step)]
            res = [pool.apply_async(self.fill_day, args=(datetime.now(),))]
            res += [pool.apply_async(self.fill_days, args=(s, e)) for s, e in date_range]
            pool.close()
            pool.join()
            for r in res:
                r.get()
        else:
            for i in range(ctx.start_days - 1, -1, -1):
                start = datetime.now() + timedelta(days=-ctx.end_days)
                self.fill_day(start + timedelta(days=-i))

    def _feature_check(self, feature_dict: dict) -> dict:
        dict_res = {k: feature_dict[k] for k in feature_dict if k in self.all_features.keys()}
        for k, v in self.all_features.items():
            if v[1] == 1:
                if v[0] == "string":
                    dict_res[k] = "<nan>" if dict_res.get(k, None) is None else str(dict_res[k])
                elif v[0] == "int64":
                    dict_res[k] = 0 if dict_res.get(k, None) is None else to_int(dict_res[k])
                else:
                    dict_res[k] = 0.0 if dict_res.get(k, None) is None else to_float(dict_res[k])
        return dict_res

    def _write(self, path: str, sql: str) -> None:
        cnt, cnt_dirty = 0, 0
        time_str = path.split('/')[-1].split('.')[0]
        with tf.io.TFRecordWriter(path) as writer:
            for row in self.data_generator(sql):
                dict_res = self.treat_one_row(row)
                if self.mode == writer_mode_debug:
                    self.debug_print(dict_res)
                else:
                    if dict_res:
                        dict_res = self._feature_check(dict_res)
                        example = convert_to_tf_example(dict_res, self.all_features)
                        writer.write(example.SerializeToString())
                        cnt += 1
                        if cnt % 10000 == 0:
                            print(f"Writing... finished {cnt} rows")
                    else:
                        cnt_dirty += 1
        if self.mode == writer_mode_debug:
            if os.path.exists(path):
                os.remove(path)
        else:
            if cnt_dirty > 0:
                print(f"Found {cnt_dirty} dirty rows, filename:{time_str}")
            if cnt <= self.threshold:
                print(f"The amount of rows in the current file is less than threshold：{cnt}.Delete!")
                os.remove(path)
            else:
                print(f"{time_str} writing success.Amount of rows:{cnt}")

    def debug_print(self, dict_res: dict) -> None:
        if not self.debug_features:
            self.debug_features = list(dict_res.keys())
        for k, v in self._feature_check(dict_res).items():
            if k in self.debug_features:
                print(f"{k}: {v}")
        print("=====================================")

    def write(self) -> None:
        if self.ctx.mode == crontab_mode:
            self.crontab()
        elif self.ctx.mode == fill_mode:
            self.fill()


@dataclass
class TrainerContext:
    data_path: str = ""
    start_days: int = 0
    online: bool = False
    test: bool = False


class BaseTrainer(object):

    def __init__(self,
                 context: TrainerContext,
                 hyper_params=None,
                 data_version="",
                 model_version="",
                 data_filter_fn=None,
                 config_root="",
                 feature_fn=None,
                 **kwargs):
        self.ctx = context
        self.hyper_params = hyper_params
        self._data_version = data_version
        self._model_version = model_version
        self.filter_fn = data_filter_fn
        self.all_features = self._dynamic_load_json()
        self._feature_list_for_distillation = list(self.all_features.keys())
        self.fc_config = self._dynamic_import_fc_config()
        self.feature_fn = feature_fn

        self.config_root = config_root

        self.train_files, self.test_files = [], []
        self.train_dataset, self.test_dataset = None, None
        self._dataset_ok = False

        self.feature_columns = MultiIODict()
        self.inputs = MultiIODict()

        self.model = None
        self.model: keras.Model

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            self._set_gpu(gpus)

    @staticmethod
    def _set_gpu(gpus: list):
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    def _dynamic_load_json(self):
        return json.load(open(f'model_features/{self._data_version}.json'))

    def _dynamic_import_fc_config(self):
        return importlib.import_module(f"{self.config_root}.{self._model_version}_config")

    def _build_feature_description(self):
        feature_description = {}
        print("start to build feature_description")
        for key, value in self.all_features.items():
            feature_description[key] = tf.io.FixedLenFeature(shape=(value[1],), dtype=TYPE_DICT[value[0]])
        return feature_description

    def _get_files(self):
        raise NotImplementedError

    def _read(self, files, feature_description, filter_fn, whether_train):
        params = {
            "filenames": [],
            "feature_description": feature_description,
            "batch_size": self.hyper_params.batch_size,
            "shuffle": whether_train,
            "labels": self.fc_config.labels
        }
        if filter_fn is not None:
            params["filter_fn"] = filter_fn
        if self.feature_fn:
            params["feature_fn"] = self.feature_fn
        if hasattr(self.fc_config, "weight_name") and whether_train:
            params["use_weight"] = True
            params["weight_name"] = self.fc_config.weight_name
        if len(files) > 0:
            params["filenames"] = files
            dataset = read_tfrecord(**params)
        else:
            dataset = None
        return dataset

    def set_dataset(self):
        self.train_dataset, self.test_dataset = self._get_dataset()
        self._dataset_ok = True
        print("dataset 构建完毕")

    def _get_dataset(self):
        if len(self.train_files) == 0:
            print("empty training data")
        feature_description = self._build_feature_description()

        train_dataset = self._read(self.train_files, feature_description, self.filter_fn, True)
        if len(self.test_files) > 0:
            test_dataset = self._read(self.test_files, feature_description, self.filter_fn, False)
        else:
            test_dataset = None
        return train_dataset, test_dataset

    def _build_fc_and_inputs(self):
        print("start to build feature_columns and inputs")
        _features = copy.deepcopy(self.fc_config.user_features)
        _features.update(self.fc_config.item_features)
        _features.update(self.fc_config.context_features)
        for key, value in _features.items():
            if key not in self._feature_list_for_distillation:
                continue
            cur, _type, _shape = None, TYPE_DICT[self.all_features[key][0]], self.all_features[key][1]
            for action in value:
                name, param = action.name, action.param
                if name is None:
                    break
                if name == 'categorical':
                    if isinstance(param, str):
                        cur = tffc.categorical_column_with_vocabulary_file(key, param, dtype=_type,
                                                                           default_value=0)
                    elif isinstance(param, list):
                        cur = tffc.categorical_column_with_vocabulary_list(key, param, dtype=_type,
                                                                           default_value=0)
                    elif isinstance(param, int):
                        cur = tffc.categorical_column_with_hash_bucket(key, hash_bucket_size=param,
                                                                       dtype=_type)
                    elif isinstance(param, tuple):
                        cur = tffc.categorical_column_with_identity(key, num_buckets=param[0], default_value=param[1])
                elif name == 'embedding':
                    cur = tffc.embedding_column(cur, dimension=param)
                elif name == 'indicator':
                    cur = tffc.indicator_column(cur)
                elif name == 'bucketize':
                    cur = tffc.bucketized_column(cur, boundaries=param)
                elif name == 'numeric':
                    cur = tffc.numeric_column(key, dtype=_type, shape=_shape)
                else:
                    print(f"unsupported feature_column type:{name}, key:{key}")
            if cur:
                self.feature_columns[key] = cur
            self.inputs[key] = keras.Input(shape=(_shape,), name=key, dtype=_type)

    @staticmethod
    def parse_optimizer(optimizer_string: str):
        """
        Only Adam is given.Add any optimizer you want.
        example:'Adam_0.001'
        """
        optimizer_dict = {"adam": keras.optimizers.Adam}
        cur = optimizer_string.lower().split("_")
        if len(cur) == 1:
            return optimizer_dict[cur[0]]()
        return optimizer_dict[cur[0]](learning_rate=float(cur[1]))

    @staticmethod
    def parse_loss_weights(weights_string: str) -> List[float]:
        # weights of task are joined by '_'
        return list(map(float, weights_string.strip().split('_')))

    def build_model(self, **kwargs) -> keras.models.Model:
        raise NotImplementedError

    def set_model(self):
        self.model = self.build_model()

    def reload(self):
        self.set_dataset()
        self.set_model()

    def set_new_hyper_params(self, new_params):
        self.hyper_params = new_params

    def prepare_data(self):
        self._get_files()
        random.shuffle(self.train_files)
        self.set_dataset()

    def reload_for_distillation(self, features_to_distill):
        self._feature_list_for_distillation = list(self.all_features.keys())
        for fea in features_to_distill:
            assert fea in self._feature_list_for_distillation, f"{fea} not in feature list"
            self._feature_list_for_distillation.remove(fea)
        self.feature_columns.clear()
        self.inputs.clear()
        self._build_fc_and_inputs()
        self.set_model()

    def train(self):
        assert self.model is not None, "model not built"
        assert self._dataset_ok is True, "dataset not prepared"

    def test(self):
        if self.test_dataset is not None:
            return self.model.evaluate(self.test_dataset())
        else:
            print("test data is empty!")
            return None


class BaseModelTestTool(object):

    def __init__(self, trainer, target_num, key_target=None):
        self.trainer = trainer
        self.target_num = target_num
        if self.target_num == 1:
            self.key_target = 1
        else:
            self.key_target = key_target
        if self.key_target is not None:
            assert self.key_target <= self.target_num, "incorrect key_target"

    @total_ordering
    @dataclass
    class Result(object):
        param_str: str
        key_index: float
        train_result: list
        test_result: list

        def __eq__(self, other):
            return self.key_index == other.key_index

        def __lt__(self, other):
            return self.key_index < other.key_index

        def __str__(self):
            return f"\nparam_str: {self.param_str}\nkey_index: {self.key_index}\ntrain_result: {self.train_result}\n" \
                   f"test_result: {self.test_result}"

    def trainer_get_result(self, param_str):
        train_result = self.trainer.train()
        test_history = self.trainer.test()
        if self.key_target is None:
            ck = 0.0
            for i in range(1, self.target_num + 1):
                ck += test_history[-i]
        else:
            ck = test_history[self.key_target - self.target_num - 1]
        test_result_list = [test_history[-i] for i in range(self.target_num, 0, -1)]

        # 如果是首页的话有 web 端测试，那就加入结果当中， 做一个参考, 不纳入核心指标计算当中
        if hasattr(self.trainer, "test_web"):
            web_history = self.trainer.test_web()
            test_result_list += [web_history[-i] for i in range(self.target_num, 0, -1)]
        return self.Result(
            param_str=param_str,
            key_index=ck,
            train_result=train_result,
            test_result=test_result_list
        )


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def to_int(x: str) -> int:
    try:
        return int(x)
    except Exception:
        return 0
