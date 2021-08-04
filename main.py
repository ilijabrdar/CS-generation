import random
import warnings

import numpy

import config
import train
import mlcq
import refactor
import models.model as svm
import preprocess


def generate_god_class_smells():
    dataset = mlcq.GodClassDS()
    samples, clean, smelly = dataset.get_original_samples()
    refactor.generate_god_class_smells(samples, clean, smelly)

    # dataset.divide_train_test_samples()
    # samples, _ = dataset.get_training_ds()
    # clean, smelly = dataset.get_total_training()
    # samples = list(samples)
    # refactor.generate_god_class_smells(samples, clean, smelly)
    refactor.extract_metrics()


def train_svm():
    dataset = mlcq.GodClassDS()
    dataset.divide_train_test_samples(0.3)
    svm.train(dataset)


if __name__ == '__main__':
    config.get_config()
    warnings.filterwarnings('ignore')
    # generate_god_class_smells()
    train.validate_and_train(model='gradient_boost', augmentation=True, sampling_strategy=0.5)
    # train_svm()

