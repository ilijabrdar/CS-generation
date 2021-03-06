import numpy as np
from dataset.mlcq import Dataset
from sklearn.model_selection import StratifiedKFold, cross_val_score
import models.model as mod
from augmentation.augmenter import SyntheticAugmenter, SMOTEAugmenter, UnOvAugmenter
import matplotlib.pyplot as plt


def validate_and_train(ds, model, augmentation=None):
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
    else:
        augmenter = __create_augmenter(augmentation, ds)
        strategy = __validate(augmenter, ds.validation_ds, model)
        print(strategy)
        augmenter.do_augmentation(strategy)
        # plt.boxplot(augmenter.augmented_ds[1])
        # plt.show()
        model = mod.train(augmenter.augmented_ds, model)
    return model


def __create_augmenter(augmentation, ds):
    if augmentation == 'synthetic':
        augmenter = SyntheticAugmenter(training_ds=ds.training_ds, generated_ds=ds.generated_ds)
    elif augmentation == 'smote':
        augmenter = SMOTEAugmenter(training_ds=ds.training_ds)
    elif augmentation == 'under_over_sampling':
        augmenter = UnOvAugmenter(training_ds=ds.training_ds)
    else:
        raise ValueError('No such augmenter')
    return augmenter


def __validate(augmenter, val_ds, clf):
    augmenter.do_augmentation(0.12)
    score_10 = __cross_validate(*augmenter.augmented_ds, *val_ds, clf)
    augmenter.do_augmentation(0.2)
    score_20 = __cross_validate(*augmenter.augmented_ds, *val_ds, clf)
    augmenter.do_augmentation(0.3)
    score_30 = __cross_validate(*augmenter.augmented_ds, *val_ds, clf)
    augmenter.do_augmentation(0.4)
    score_40 = __cross_validate(*augmenter.augmented_ds, *val_ds, clf)
    augmenter.do_augmentation(0.5)
    score_50 = __cross_validate(*augmenter.augmented_ds, *val_ds, clf)
    scores = {0.12: score_10, 0.2: score_20, 0.3: score_30, 0.4: score_40, 0.5: score_50}
    print(scores)
    return max(scores, key=scores.get)


def __load_datasets(ds, augmentation):
    __load_original_dataset(ds)
    if augmentation is not None and augmentation == 'synthetic':
        __load_generated_dataset(ds)


def __load_original_dataset(ds: Dataset):
    ds.divide_train_test_samples()
    print('original: ', __count_ds(ds.ds))
    print('training: ', __count_ds(ds.training_ds))
    print('validation: ', __count_ds(ds.validation_ds))
    print('test: ', __count_ds(ds.test_ds))


def __load_generated_dataset(ds: Dataset):
    ds.load_and_preprocess_generated_ds()
    print('generated: ', __count_ds(ds.generated_ds))


def __cross_validate(train_x, train_y, val_x, val_y, model):
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # scores = cross_val_score(model, train_x, train_y, cv=skf, scoring='f1')
    # avg = round(sum(scores) / len(scores), 2)
    trained_model = mod.train((train_x, train_y), model)
    return mod.test(trained_model, (val_x, val_y))


def __count_ds(ds):
    flat = np.ravel(ds[1])
    unique, counts = np.unique(flat, return_counts=True)
    total = dict(zip(unique, counts))
    if 0 not in total:
        total[0] = 0
    if 1 not in total:
        total[1] = 0
    return total[0], total[1]

