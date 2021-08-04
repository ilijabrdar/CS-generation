import operator

import config
import numpy as np
from mlcq import GodClassDS
from sklearn.model_selection import StratifiedKFold, cross_val_score
import models.svm as svm


def validate_and_train(model, augmentation=False, sampling_strategy=0.3):
    ds = GodClassDS()
    __load_datasets(ds)
    if model == 'svm':
        clf = svm.create_svm()
        if augmentation:
            strategy = __validate(ds, clf)
            print(strategy)
            ds.augment_ds(strategy)
            clf = svm.train(ds.augmented_ds, clf)
        else:
            clf = svm.train(ds.training_ds, clf)
        svm.test(clf, ds.test_ds)


def __validate(ds, clf):
    # clf = svm.create_svm()
    ds.augment_ds(0.1)
    score_10 = __cross_validate(ds.augmented_ds[0], ds.augmented_ds[1], clf)
    ds.augment_ds(0.2)
    score_20 = __cross_validate(ds.augmented_ds[0], ds.augmented_ds[1], clf)
    ds.augment_ds(0.3)
    score_30 = __cross_validate(ds.augmented_ds[0], ds.augmented_ds[1], clf)
    ds.augment_ds(0.4)
    score_40 = __cross_validate(ds.augmented_ds[0], ds.augmented_ds[1], clf)
    ds.augment_ds(0.5)
    score_50 = __cross_validate(ds.augmented_ds[0], ds.augmented_ds[1], clf)
    scores = {0.1: score_10, 0.2: score_20, 0.3: score_30, 0.4: score_40, 0.5: score_50}
    print(scores)
    return max(scores, key=scores.get)


def __load_datasets(ds):
    __load_original_dataset(ds)
    __load_generated_dataset(ds)


def __load_original_dataset(ds: GodClassDS):
    ds.divide_train_test_samples()
    print('original: ', count_ds(ds.ds))
    print('training: ', count_ds(ds.training_ds))
    print('test: ', count_ds(ds.test_ds))


def __load_generated_dataset(ds: GodClassDS):
    ds.load_and_preprocess_generated_ds()
    print('generated: ', count_ds(ds.generated_ds))


def __cross_validate(train_x, train_y, model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, train_x, train_y, cv=skf, scoring='f1')
    avg = round(sum(scores) / len(scores), 2)
    return avg


def count_ds(ds):
    flat = np.ravel(ds[1])
    unique, counts = np.unique(flat, return_counts=True)
    total = dict(zip(unique, counts))
    if 0 not in total:
        total[0] = 0
    if 1 not in total:
        total[1] = 0
    return total[0], total[1]

