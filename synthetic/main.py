import warnings

from dataset import mlcq
import refactor
from util import config

GC_DATASET_PATH = config.get('DATASET', 'GC_DIR_PATH')
LM_DATASET_PATH = config.get('DATASET', 'LM_DIR_PATH')
LM_WRAP_DATASET_PATH = config.get('DATASET', 'LM_WRAPPED_DIR_PATH')

GC_OUTPUT = config.get('GOD_CLASS_TRAIN', 'OUTPUT_DIR_PATH')
GC_SAMPLING_RATIO = config.get('GOD_CLASS_TRAIN', 'SAMPLING_RATIO')
GC_MAX_ITERATIONS = config.get('GOD_CLASS_TRAIN', 'MAX_ITERATIONS')

LM_OUTPUT = config.get('LONG_METHOD_TRAIN', 'OUTPUT_DIR_PATH')
LM_SAMPLING_RATIO = config.get('LONG_METHOD_TRAIN', 'SAMPLING_RATIO')
LM_MAX_ITERATIONS = config.get('LONG_METHOD_TRAIN', 'MAX_ITERATIONS')

CK_GC_OUTPUT = config.get('CK_SERVICE', 'GC_OUTPUT_DIR_PATH')
CK_LM_OUTPUT = config.get('CK_SERVICE', 'LM_OUTPUT_DIR_PATH')


def generate_god_class_smells():
    dataset = mlcq.Dataset(GC_DATASET_PATH)
    samples, clean, smelly = dataset.get_original_samples()
    refactor.generate_smells(samples=samples, smelly=smelly, clean=clean, sampling_ratio=GC_SAMPLING_RATIO,
                             output=GC_OUTPUT, max_iterations=GC_MAX_ITERATIONS, smell_type='god_class')
    refactor.extract_metrics(metrics_output=CK_GC_OUTPUT, original_dataset_path=GC_DATASET_PATH,
                             generated_dataset_path=GC_OUTPUT)


def generate_long_method_smells():
    dataset = mlcq.Dataset(LM_DATASET_PATH)
    dataset.wrap_in_class(LM_WRAP_DATASET_PATH)
    samples, clean, smelly = dataset.get_original_samples()
    refactor.generate_smells(samples=samples, smelly=smelly, clean=clean, sampling_ratio=LM_SAMPLING_RATIO,
                             output=LM_OUTPUT, max_iterations=LM_MAX_ITERATIONS, smell_type='long_method')
    refactor.extract_metrics(metrics_output=CK_LM_OUTPUT, original_dataset_path=LM_WRAP_DATASET_PATH,
                             generated_dataset_path=LM_OUTPUT)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # generate_god_class_smells()
    generate_long_method_smells()

