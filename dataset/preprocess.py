import numpy as np
import pandas as pd
import os


def gc_preprocess_data(target_path: str):
    samples = __get_all_class_samples(target_path)
    samples = __drop_inner_classes(samples)
    if target_path.endswith('original'):
        labels = __extract_labels(samples)
    else:
        labels = [1] * samples.shape[0]
    samples = __gc_filter_samples(samples)
    samples = __min_max_normalization(samples)
    return __transform_to_np_arrays(samples, labels)


def lm_preprocess_data(target_path: str):
    samples = __get_all_method_samples(target_path)
    samples = __from_anonymous_methods(samples)
    if target_path.endswith('original'):
        labels = __extract_labels(samples)
    else:
        labels = [1] * samples.shape[0]
    samples = __lm_filter_samples(samples)
    samples = __min_max_normalization(samples)
    return __transform_to_np_arrays(samples, labels)


def __get_all_class_samples(path):
    return pd.read_csv(os.path.join(path, 'class.csv'))


def __get_all_method_samples(path):
    return pd.read_csv(os.path.join(path, 'method.csv'))


def __extract_labels(samples: pd.DataFrame):
    labels = []
    files = samples['file'].tolist()
    for file in files:
        if file.split(os.path.sep)[-2] == 'none':
            labels.append(0)
        else:
            labels.append(1)
    return labels


def __drop_inner_classes(samples: pd.DataFrame):
    is_class_or_interface = (samples['type'] == 'class') | (samples['type'] == 'interface')
    return samples[is_class_or_interface]


def __from_anonymous_methods(samples: pd.DataFrame):
    is_not_anonymous = ~samples['class'].str.contains('Anonymous')
    return samples[is_not_anonymous]


def __gc_filter_samples(samples: pd.DataFrame):
    samples = samples.drop(columns=['file', 'class', 'type', 'cbo', 'dit', 'rfc', 'tcc', 'lcc', 'nosi'])
    return samples


def __lm_filter_samples(samples: pd.DataFrame):
    samples = samples.drop(columns=['file', 'class', 'method', 'constructor', 'line',  'cbo', 'rfc', 'hasJavaDoc'])
    return samples


def __min_max_normalization(samples: pd.DataFrame):
    for col in samples.columns:
        samples[col] = (samples[col] - samples[col].min()) / (samples[col].max() - samples[col].min())
    samples = samples.fillna(0)
    return samples


def __transform_to_np_arrays(samples: pd.DataFrame, labels):
    return samples.to_numpy(), np.reshape(labels, (-1, 1))
