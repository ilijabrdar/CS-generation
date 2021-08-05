from sklearn.model_selection import train_test_split
import os
import numpy as np
from util import config
from dataset import preprocess

CK_OUTPUT_ORIGINAL = os.path.join(config.get('CK_SERVICE', 'OUTPUT_DIR_PATH'), 'original')
CK_OUTPUT_GENERATED = os.path.join(config.get('CK_SERVICE', 'OUTPUT_DIR_PATH'), 'generated')


class GodClassDS:
    def __init__(self):
        self.dataset_path = config.get('DATASET', 'DIR_PATH')
        self.ds = ([], [])
        self.training_ds = ([], [])
        self.test_ds = ([], [])
        self.generated_ds = ([], [])
        self.augmented_ds = ([], [])

    def get_original_samples(self):
        total = 0
        clean = 0
        files = []
        for root, directory, filenames in os.walk(self.dataset_path):
            total += len(filenames)
            if root.endswith('none'):
                clean = len(filenames)
                for filename in filenames:
                    files.append(os.path.join(root, filename))
        return files, clean, total - clean

    def load_and_preprocessed_original_ds(self):
        samples, labels = preprocess.preprocess_data(CK_OUTPUT_ORIGINAL)
        self.ds = (samples, labels)

    def load_and_preprocess_generated_ds(self):
        samples, labels = preprocess.preprocess_data(CK_OUTPUT_GENERATED)
        self.generated_ds = (samples, labels)

    def divide_train_test_samples(self):
        self.load_and_preprocessed_original_ds()
        x, y = self.ds
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
        self.training_ds = (x_train, y_train)
        self.test_ds = (x_test, y_test)

    def augment_ds(self, sampling_strategy):
        clean, smelly = self.count_training_ds()
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

    def get_training_ds(self):
        return self.training_ds

    def get_test_ds(self):
        return self.test_ds

    def count_training_ds(self):
        flat = np.ravel(self.training_ds[1])
        unique, counts = np.unique(flat, return_counts=True)
        total = dict(zip(unique, counts))
        return total[0], total[1]