import numpy as np
import pandas as pd
import os


def preprocess_data(target_path: str):
    samples = __get_all_samples_and_labels(target_path)
    samples = __drop_inner_classes(samples)
    if target_path.endswith('original'):
        labels = __extract_labels(samples)
    else:
        labels = [1] * samples.shape[0]
    samples = __filter_samples(samples)
    samples = __min_max_normalization(samples)
    return __transform_to_np_arrays(samples, labels)


def __get_all_samples_and_labels(path):
    return pd.read_csv(os.path.join(path, 'class.csv'))


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


def __filter_samples(samples: pd.DataFrame):
    samples = samples.drop(columns=['file', 'class', 'type', 'cbo', 'dit', 'rfc', 'tcc', 'lcc', 'nosi'])
    return samples


def __min_max_normalization(samples: pd.DataFrame):
    for col in samples.columns:
        samples[col] = (samples[col] - samples[col].min()) / (samples[col].max() - samples[col].min())
    samples = samples.fillna(0)
    return samples


def __transform_to_np_arrays(samples: pd.DataFrame, labels):
    return samples.to_numpy(), np.reshape(labels, (-1, 1))
