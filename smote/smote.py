from imblearn.over_sampling import SMOTE
from collections import Counter


def do_smote_augmentation(ds):
    sm = SMOTE(random_state=42)
    x_res, y_res = sm.fit_resample(ds[0], ds[1])
    print('Resampled dataset shape %s' % Counter(y_res))

