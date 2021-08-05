import numpy as np
from dataset.mlcq import GodClassDS
from sklearn.model_selection import StratifiedKFold, cross_val_score
import models.model as mod
from augmentation.augmenter import SyntheticAugmenter


def validate_and_train(model, augmentation=None):
    ds = GodClassDS()
    __load_datasets(ds, augmentation)
    if model == 'svm':
        clf = mod.create_svm()
        clf = __train_model(augmentation, clf, ds)
        mod.test(clf, ds.test_ds)
    elif model == 'random_forest':
        rf = mod.create_rf()
        rf = __train_model(augmentation, rf, ds)
        mod.test(rf, ds.test_ds)
    elif model == 'gradient_boost':
        gb = mod.create_gb()
        gb = __train_model(augmentation, gb, ds)
        mod.test(gb, ds.test_ds)
    elif model == 'bagging':
        bg = mod.create_bagging()
        bg = __train_model(augmentation, bg, ds)
        mod.test(bg, ds.test_ds)
    else:
        raise ValueError('No such model')


def __train_model(augmentation, model, ds):
    if augmentation is None:
        model = mod.train(ds.training_ds, model)
    elif augmentation == 'synthetic':
        sa = SyntheticAugmenter(training_ds=ds.training_ds, generated_ds=ds.generated_ds)
        strategy = __validate(sa, model)
        print(strategy)
        sa.do_augmentation(strategy)
        model = mod.train(sa.augmented_ds, model)

    return model


def __validate(augmenter, clf):
    augmenter.do_augmentation(0.1)
    score_10 = __cross_validate(augmenter.augmented_ds[0], augmenter.augmented_ds[1], clf)
    augmenter.do_augmentation(0.2)
    score_20 = __cross_validate(augmenter.augmented_ds[0], augmenter.augmented_ds[1], clf)
    augmenter.do_augmentation(0.3)
    score_30 = __cross_validate(augmenter.augmented_ds[0], augmenter.augmented_ds[1], clf)
    augmenter.do_augmentation(0.4)
    score_40 = __cross_validate(augmenter.augmented_ds[0], augmenter.augmented_ds[1], clf)
    augmenter.do_augmentation(0.5)
    score_50 = __cross_validate(augmenter.augmented_ds[0], augmenter.augmented_ds[1], clf)
    scores = {0.1: score_10, 0.2: score_20, 0.3: score_30, 0.4: score_40, 0.5: score_50}
    print(scores)
    return max(scores, key=scores.get)


def __load_datasets(ds, augmentation):
    __load_original_dataset(ds)
    if augmentation is not None and augmentation == 'synthetic':
        __load_generated_dataset(ds)


def __load_original_dataset(ds: GodClassDS):
    ds.divide_train_test_samples()
    print('original: ', __count_ds(ds.ds))
    print('training: ', __count_ds(ds.training_ds))
    print('test: ', __count_ds(ds.test_ds))


def __load_generated_dataset(ds: GodClassDS):
    ds.load_and_preprocess_generated_ds()
    print('generated: ', __count_ds(ds.generated_ds))


def __cross_validate(train_x, train_y, model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, train_x, train_y, cv=skf, scoring='f1')
    avg = round(sum(scores) / len(scores), 2)
    return avg


def __count_ds(ds):
    flat = np.ravel(ds[1])
    unique, counts = np.unique(flat, return_counts=True)
    total = dict(zip(unique, counts))
    if 0 not in total:
        total[0] = 0
    if 1 not in total:
        total[1] = 0
    return total[0], total[1]

