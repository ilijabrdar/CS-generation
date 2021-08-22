import warnings

import dataset.train as train
from dataset.mlcq import Dataset
from util import config

GC_DATASET_PATH = config.get('DATASET', 'GC_DIR_PATH')
LM_WRAP_DATASET_PATH = config.get('DATASET', 'LM_WRAPPED_DIR_PATH')


def test_god_class():
    ds = Dataset(GC_DATASET_PATH, 'god_class')
    train.validate_and_train(ds, 'bagging', augmentation='smote')


def test_long_method():
    ds = Dataset(GC_DATASET_PATH, 'long_method')
    train.validate_and_train(ds, 'bagging', augmentation='synthetic')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    test_long_method()

