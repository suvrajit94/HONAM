import numpy as np
from sklearn.linear_model import LogisticRegression

from load_data import load_pima_indians_diabetes_database
from preprocessor import Preprocessor
from utils import evaluate

task = 'binary_classification'


def run_single():
    _, (x_train, y_train), (_, _), (x_test, y_test) = load_pima_indians_diabetes_database()

    preprocessor = Preprocessor(task=task, n_quantiles=x_train.shape[0])
    x_train, y_train = preprocessor.fit_transform(x_train, y_train)
    x_test, y_test = preprocessor.transform(x_test, y_test)

    # train
    model = LogisticRegression()
    model.fit(x_train, y_train.ravel())

    # test
    y_hat = model.predict_proba(x_test)[:, 1]

    # evaluate
    auroc, auprc = evaluate(y_test, y_hat, task)
    return auroc, auprc


def run_multiple(n=5):
    aurocs = []
    auprcs = []
    for i in range(n):
        auroc, auprc = run_single()
        aurocs.append(auroc)
        auprcs.append(auprc)

    auroc_mean = np.mean(aurocs)
    auroc_std = np.std(aurocs)
    auprc_mean = np.mean(auprcs)
    auprc_std = np.std(auprcs)

    return (auroc_mean, auroc_std), (auprc_mean, auprc_std)


if __name__ == '__main__':
    (auroc_mean, auroc_std), (auprc_mean, auprc_std) = run_multiple()
    print(f'AUROC: {auroc_mean:.7f} (+/- {auroc_std:.7f})')
    print(f'AUPRC: {auprc_mean:.7f} (+/- {auprc_std:.7f})')

