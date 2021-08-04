import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def evaluate(predicted, correct):
    predicted = np.ravel(predicted).tolist()
    correct = np.ravel(correct).tolist()
    accuracy = accuracy_score(correct, predicted)
    f_measure = f1_score(correct, predicted, zero_division=0)
    print('accuracy: ', accuracy)
    print('f-measure: ', f_measure)
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(predicted)):
        if predicted[i] == 1 and correct[i] == 1:
            tp += 1
        elif predicted[i] == 1 and correct[i] == 0:
            fp += 1
        elif predicted[i] == 0 and correct[i] == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f = 2 * precision * recall / (precision + recall)
    acc = (tp + tn) / (tp + tn + fp + fn)
    print('precision', precision)
    print('recall', recall)
    print('f', f)
    print('acc', acc)
    return f_measure

