import warnings

from dataset import mlcq
import refactor


def generate_god_class_smells():
    dataset = mlcq.GodClassDS()
    samples, clean, smelly = dataset.get_original_samples()
    refactor.generate_god_class_smells(samples, clean, smelly)
    refactor.extract_metrics()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    generate_god_class_smells()
