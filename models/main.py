import warnings

import dataset.train as train

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    train.validate_and_train('bagging', augmentation='under_over_sampling')

