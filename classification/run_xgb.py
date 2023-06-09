import numpy as np
from xgboost import XGBClassifier

from load_data import load_pima_indians_diabetes_database
from preprocessor import Preprocessor
from utils import evaluate

task = 'binary_classification'


def run_single():
    _, (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_pima_indians_diabetes_database()

    preprocessor = Preprocessor(task=task, n_quantiles=x_train.shape[0])
    x_train, y_train = preprocessor.fit_transform(x_train, y_train)
    x_val, y_val = preprocessor.transform(x_val, y_val)
    x_test, y_test = preprocessor.transform(x_test, y_test)
    x_train = np.concatenate((x_train, x_val))
    y_train = np.concatenate((y_train, y_val)).ravel()

    # train
    model = MLPClassifier(
        hidden_layer_sizes=(32,64,32),
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        tol=0.001,
        alpha=0.0001)
    model.fit(x_train, y_train)

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