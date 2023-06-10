import numpy as np
from deepctr_torch.inputs import DenseFeat
from deepctr_torch.models import DCN

from load_data import load_pima_indians_diabetes_database
from preprocessor import Preprocessor
from utils import evaluate

task = 'binary_classification'


def run_single():
    columns, (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_pima_indians_diabetes_database()

    preprocessor = Preprocessor(task=task, n_quantiles=x_train.shape[0])
    x_train, y_train = preprocessor.fit_transform(x_train, y_train)
    x_val, y_val = preprocessor.transform(x_val, y_val)
    x_test, y_test = preprocessor.transform(x_test, y_test)
    x_train = np.concatenate((x_train, x_val))
    y_train = np.concatenate((y_train, y_val)).ravel()

    feature_columns = [DenseFeat(col, 1) for col in columns]
    # train
    model = DCN(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns, dnn_hidden_units=[16, 32, 16], task='binary', device="cpu", l2_reg_embedding=1e-02, l2_reg_cross=1e-02, l2_reg_dnn=0.5, dnn_dropout=0.5)
    model.compile("adam", "mse", metrics=['mse'])

    train_model_input = dict(zip(columns, x_train.T))
    val_model_input = dict(zip(columns, x_val.T))
    test_model_input = dict(zip(columns, x_test.T))

    model.fit(train_model_input, y_train)

    # test
    y_hat = model.predict(test_model_input)

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