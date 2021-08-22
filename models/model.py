from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from models import evaluate


def create_svm():
    return svm.SVC(kernel='rbf', C=0.2, random_state=0)


def create_rf():
    return RandomForestClassifier(max_depth=5, random_state=0)


def create_gb():
    return GradientBoostingClassifier()


def create_bagging():
    svc = svm.SVC(C=0.1, kernel='linear', gamma='scale')
    return BaggingClassifier(base_estimator=svc, bootstrap=True, random_state=0)


def train(ds, model):
    features, labels = ds
    model.fit(features, labels)
    return model


def test(model, ds):
    features, labels = ds
    predicted_labels = model.predict(features)
    return evaluate.evaluate(predicted_labels, labels)

