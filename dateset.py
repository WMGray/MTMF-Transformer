from collections import Counter
from typing import Tuple, Dict, Any, List

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skorch.helper import SliceDict
from utils import setup_seed, check_sample_order


class dataset(Dataset):
    def __init__(self, path: str, use_cols: List = None, normalize: bool = True):
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


def load_uni_features(seed: int, disease: str, feature: str) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    """
    :param feature: type of feature: ko or species
    :param seed: random seed for train and test split
    :param disease: prefix of dataset to open
    :return:
    """
    feature = feature.split(',')
    print(feature)
    path = f"/hdd/wmh/Disease/Data/{disease}/{feature[0]}_abundance.csv"
    data = dataset(path, use_cols=None)  # Z-Score

    # 划分数据
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.label.astype('int'),
                                                        test_size=0.2,
                                                        random_state=seed,
                                                        stratify=data.label)
    # 合并两个输入 -- Skorch
    x_train = SliceDict(f1_input=x_train.astype(np.float32))
    x_test = SliceDict(f1_input=x_test.astype(np.float32))

    y_train, y_test = y_train, y_test
    print(Counter(y_train), Counter(y_test))

    y_train = np.expand_dims(y_train, axis=1).astype(np.float32)

    return x_train, x_test, y_train, y_test


def load_multi_features(seed: int, disease: str, feature: str) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    """
    :param feature: type of feature: ko or species
    :param seed: random seed for train and test split
    :param disease: prefix of dataset to open
    :return:
    """
    feature = feature.split(',')
    print(feature)
    f1_path = f"./Data/{disease}/{feature[0]}_abundance.csv"
    f2_path = f"./Data/{disease}/{feature[1]}_abundance.csv"

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
