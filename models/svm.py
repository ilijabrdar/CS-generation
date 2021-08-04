from sklearn import svm
import evaluate


def create_svm():
    return svm.SVC(kernel='linear')


def train(ds, clf):
    features, labels = ds
    clf.fit(features, labels)
    return clf


def test(clf, ds):
    features, labels = ds
    predicted_labels = clf.predict(features)
    evaluate.evaluate(predicted_labels, labels)

