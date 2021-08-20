import warnings

import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def evaluate(predicted, correct):
    predicted = np.ravel(predicted).tolist()
    correct = np.ravel(correct).tolist()
    f_measure = f1_score(correct, predicted, zero_division=0)
    print('f-measure: ', f_measure)
    return f_measure

