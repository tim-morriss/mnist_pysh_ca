import numpy as np
from load_datasets import LoadDatasets
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier


def run():
    X, y = make_multilabel_classification(random_state=0)
    print(X)
    print(y)

    inner_clf = LogisticRegression(solver="liblinear", random_state=0)
    clf = MultiOutputClassifier(inner_clf).fit(X, y)

    y_score = np.transpose([y_pred[:, 1] for y_pred in clf.predict_proba(X)])
    print(y_score)

    print(roc_auc_score(y, y_score, average=None))


if __name__ == '__main__':
    run()
