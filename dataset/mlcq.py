from sklearn.model_selection import train_test_split
import os
import numpy as np
from collections import Counter
from util import config
from dataset import preprocess
import synthetic.main as syn

CK_GC_OUTPUT_ORIGINAL = os.path.join(config.get('CK_SERVICE', 'GC_OUTPUT_DIR_PATH'), 'original')
CK_GC_OUTPUT_GENERATED = os.path.join(config.get('CK_SERVICE', 'GC_OUTPUT_DIR_PATH'), 'generated')

CK_LM_OUTPUT_ORIGINAL = os.path.join(config.get('CK_SERVICE', 'LM_OUTPUT_DIR_PATH'), 'original')
CK_LM_OUTPUT_GENERATED = os.path.join(config.get('CK_SERVICE', 'LM_OUTPUT_DIR_PATH'), 'generated')

LM_WRAP_DATASET_PATH = config.get('DATASET', 'LM_WRAPPED_DIR_PATH')


class Dataset:
    def __init__(self, path, ds_type='god_class'):
        self.dataset_path = path
        self.dataset_type = ds_type
        self.ds = ([], [])
        self.ds_files = ([], [], [])
        self.training_ds = ([], [])
        self.test_ds = ([], [])
        self.validation_ds = ([], [])
        self.generated_ds = ([], [])

    def wrap_in_class(self, output_path):
        if os.path.exists(output_path):
            self.dataset_path = output_path
            return
        os.mkdir(output_path)
        os.chdir(output_path)
        for _dir in os.listdir(self.dataset_path):
            os.mkdir(_dir)
            for file in os.listdir(os.path.join(self.dataset_path, _dir)):
                with open(os.path.join(self.dataset_path, _dir, file), 'r') as f:
                    file_content = f.read()
                with open(os.path.join(output_path, _dir, file), 'w') as f:
                    f.write('public class Name {' + file_content + '}')

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
        if self.dataset_type == 'god_class':
            syn.process_original_ds('god_class')
            samples, labels = preprocess.gc_preprocess_data(CK_GC_OUTPUT_ORIGINAL)
        else:
            self.wrap_in_class(LM_WRAP_DATASET_PATH)
            syn.process_original_ds('long_method')
            samples, labels = preprocess.lm_preprocess_data(CK_LM_OUTPUT_ORIGINAL)
        self.ds = (samples, labels)

    def load_and_preprocess_generated_ds(self):
        smelly, clean = self.__count_original_ds()
        if self.dataset_type == 'god_class':
            syn.generate_god_class_smells(self.ds_files[0], clean, smelly)
            samples, labels = preprocess.gc_preprocess_data(CK_GC_OUTPUT_GENERATED)
        else:
            syn.generate_long_method_smells(self.ds_files[0], clean, smelly)
            samples, labels = preprocess.lm_preprocess_data(CK_LM_OUTPUT_GENERATED)
        self.generated_ds = (samples[:, 1:], labels)

    def divide_train_test_samples(self):
        self.load_and_preprocessed_original_ds()
        x, y = self.ds
        x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=42,
                                                          stratify=y_train_val)
        self.ds_files = (x_train[:, 0].tolist(), x_val[:, 0].tolist(), x_test[:, 0].tolist())
        self.training_ds = (x_train[:, 1:], y_train)
        self.validation_ds = (x_val[:, 1:], y_val)
        self.test_ds = (x_test[:, 1:], y_test)

    def __count_original_ds(self):
        labels = np.ravel(self.ds[1])
        cnt = Counter(labels)
        return cnt[1], cnt[0]
