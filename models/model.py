from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from models import evaluate


def create_svm():
    return svm.SVC(kernel='linear')


def create_rf():
    return RandomForestClassifier(max_depth=5, random_state=0)


def create_gb():
    return GradientBoostingClassifier()


def create_bagging():
    return BaggingClassifier(base_estimator=svm.SVC(), n_estimators=10, random_state=0)


def train(ds, model):
    features, labels = ds
    model.fit(features, labels)
    return model


def test(model, ds):
    features, labels = ds
    predicted_labels = model.predict(features)
    evaluate.evaluate(predicted_labels, labels)

