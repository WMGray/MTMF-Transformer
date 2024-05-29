from collections import Counter
import os
from os.path import join as join
from typing import Tuple, Dict, Any

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skorch.helper import SliceDict
#########################################
# 待修改
from utils import setup_seed, check_sample_order


class dataset(Dataset):
    def __init__(self, path: str, use_cols=None, normalize=True):
        super().__init__()
        if use_cols is None:
            use_cols = []
        print(f'Load dataset from {path}')

        if use_cols:
            self.data = pd.read_csv(path)[use_cols]
        else:
            self.data = pd.read_csv(path)
        # self.data.sort_values('sample_id', inplace=True)  # sort

        self.label = self.data.iloc[:, 1].values.squeeze()
        self.data = self.data.iloc[:, 2:].values

        if normalize:
            scaler = StandardScaler()
            self.data = scaler.fit_transform(self.data)

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

def load_single_features(seed: int, disease: str, feature: str) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    """
    :param seed: random seed for train and test split
    :param disease: prefix of dataset to open
    :return:
    """
    feature = feature.split(',')
    print(feature)
    path = f"/hdd/wmh/Disease/Data/{disease}/{feature[0]}_abundance.csv"


    data = dataset(path, use_cols=None)  # Z-Score

    # 划分数据
    x_train_ko, x_test_ko, y_train_ko, y_test_ko = train_test_split(data.data, data.label.astype('int'),
                                                                    test_size=0.2,
                                                                    random_state=seed,
                                                                    stratify=data.label)
    # 合并两个输入 -- Skorch
    x_train = SliceDict(f1_input=x_train_ko.astype(np.float32))
    x_test = SliceDict(f1_input=x_test_ko.astype(np.float32))

    y_train, y_test = y_train_ko, y_test_ko
    print(Counter(y_train), Counter(y_test))

    y_train = np.expand_dims(y_train, axis=1).astype(np.float32)

    return x_train, x_test, y_train, y_test


def load_full_features(seed: int, disease: str, feature: str) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    """
    :param seed: random seed for train and test split
    :param disease: prefix of dataset to open
    :return:
    """
    feature = feature.split(',')
    print(feature)
    f1_path = f"/hdd/wmh/Disease/Data/{disease}/{feature[0]}_abundance.csv"
    f2_path = f"/hdd/wmh/Disease/Data/{disease}/{feature[1]}_abundance.csv"

    check_sample_order([f1_path, f2_path])

    f1_data = dataset(f1_path, use_cols=None)  # Z-Score
    f2_data = dataset(f2_path, use_cols=None)

    # 划分数据
    x_train_ko, x_test_ko, y_train_ko, y_test_ko = train_test_split(f1_data.data, f1_data.label.astype('int'),
                                                                    test_size=0.2,
                                                                    random_state=seed,
                                                                    stratify=f1_data.label)
    x_train_go, x_test_go, y_train_go, y_test_go = train_test_split(f2_data.data, f2_data.label.astype('int'),
                                                                    test_size=0.2,
                                                                    random_state=seed,
                                                                    stratify=f2_data.label)
    # 合并两个输入 -- Skorch
    if (y_train_ko.all() == y_train_go.all()) and (y_test_ko.all() == y_test_go.all()):
        x_train = SliceDict(f1_input=x_train_ko.astype(np.float32), f2_input=x_train_go.astype(np.float32))
        x_test = SliceDict(f1_input=x_test_ko.astype(np.float32), f2_input=x_test_go.astype(np.float32))

        y_train, y_test = y_train_ko, y_test_ko
        print(Counter(y_train), Counter(y_test))

        y_train = np.expand_dims(y_train, axis=1).astype(np.float32)

        return x_train, x_test, y_train, y_test
    else:
        assert 0, "两个特征的标签不匹配"


def load_select_full_features(seed: int, disease: str) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    """
    :param seed: random seed for train and test split
    :param disease: prefix of dataset to open
    :return:
    """
    ko_path = f"/hdd/wmh/Disease/Data/{disease}/ko_abundance_select_{seed}.csv"
    go_path = f"/hdd/wmh/Disease/Data/{disease}/go_abundance_select_{seed}.csv"

    check_sample_order([ko_path, go_path])

    ko_data = dataset(ko_path, use_cols=None)  # Z-Score
    go_data = dataset(go_path, use_cols=None)

    # 划分数据
    x_train_ko, x_test_ko, y_train_ko, y_test_ko = train_test_split(ko_data.data, ko_data.label.astype('int'),
                                                                    test_size=0.2,
                                                                    random_state=seed,
                                                                    stratify=ko_data.label)
    x_train_go, x_test_go, y_train_go, y_test_go = train_test_split(go_data.data, go_data.label.astype('int'),
                                                                    test_size=0.2,
                                                                    random_state=seed,
                                                                    stratify=go_data.label)
    # 合并两个输入 -- Skorch
    if (y_train_ko.all() == y_train_go.all()) and (y_test_ko.all() == y_test_go.all()):
        x_train = SliceDict(ko_input=x_train_ko.astype(np.float32), go_input=x_train_go.astype(np.float32))
        x_test = SliceDict(ko_input=x_test_ko.astype(np.float32), go_input=x_test_go.astype(np.float32))

        y_train, y_test = y_train_ko, y_test_ko
        print(Counter(y_train), Counter(y_test))

        y_train = np.expand_dims(y_train, axis=1).astype(np.float32)

        return x_train, x_test, y_train, y_test
    else:
        assert 0, "两个特征的标签不匹配"


def load_split_features(disease: str, feature: str):
    """
    :param disease: prefix of dataset to open
    :param feature: type of feature to use, eg: emo,egemaps,
    :return:
    """
    if disease != "AD":
        assert 0
    global y_test, y_train
    dir_dict = {
        'emo': "/hdd/wmh/Disease/Data/AD/emo_large/",
        'egemaps': "/hdd/wmh/Disease/Data/AD/eGeMAPSv02/",
        'compare': "/hdd/wmh/Disease/Data/AD/ComParE_2016/",
        'liwc': "/hdd/wmh/Disease/Data/AD/linguistic/"
    }
    use_features = [x.lower().strip() for x in feature.split(",")]
    use_features = {k: dir_dict[k] for k in use_features}

    x_train, x_test = SliceDict(), SliceDict()
    for name, path in use_features.items():
        train_path, test_path = path + "train.csv", path + "test.csv"

        train_data = dataset(train_path, use_cols=None)  # Z-Score
        test_data = dataset(test_path, use_cols=None)

        x_train[name] = train_data.data.astype(np.float32)
        x_test[name] = test_data.data.astype(np.float32)

        y_train, y_test = train_data.label, test_data.label
        y_train = np.expand_dims(y_train, axis=1).astype(np.float32)

    return x_train, x_test, y_train, y_test
