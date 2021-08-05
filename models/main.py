import warnings

import dataset.train as train

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    train.validate_and_train('svm', augmentation='synthetic')

