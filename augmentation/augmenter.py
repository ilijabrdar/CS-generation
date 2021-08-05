import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter


class Augmenter:
    def __init__(self, ds):
        self.training_ds = ds
        self.augmented_ds = ([], [])

    def do_augmentation(self, sampling_strategy=0.3):
        self.augmented_ds = self.training_ds


class SyntheticAugmenter(Augmenter):
    def __init__(self, training_ds, generated_ds):
        super().__init__(training_ds)
        self.generated_ds = generated_ds

    def do_augmentation(self, sampling_strategy=0.3):
        clean, smelly = self.__count_training_ds()
        rows_num = self.generated_ds[0].shape[0]
        delta = round(clean * sampling_strategy / (1 - sampling_strategy) - smelly)
        np.random.seed(500)
        if delta > 0:
            random_ind = np.random.choice(rows_num, size=delta, replace=False)
            to_add_feat = self.generated_ds[0][random_ind, :]
            to_add_lbl = self.generated_ds[1][random_ind, :]
            aug_feat = np.concatenate((self.training_ds[0], to_add_feat))
            aug_lbl = np.concatenate((self.training_ds[1], to_add_lbl))
        else:
            aug_feat, aug_lbl = self.training_ds
        self.augmented_ds = (aug_feat, aug_lbl)

    def __count_training_ds(self):
        flat = np.ravel(self.training_ds[1])
        unique, counts = np.unique(flat, return_counts=True)
        total = dict(zip(unique, counts))
        return total[0], total[1]


class SMOTEAugmenter(Augmenter):
    def __init__(self, training_ds):
        super().__init__(training_ds)

    def do_augmentation(self, sampling_strategy=0.3):
        ratio = sampling_strategy / (1 - sampling_strategy)
        sm = SMOTE(random_state=42, sampling_strategy=ratio, k_neighbors=200)
        self.augmented_ds = sm.fit_resample(self.training_ds[0], self.training_ds[1])

