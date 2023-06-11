import numpy as np
from deepctr_torch.inputs import DenseFeat
from deepctr_torch.models import DCN

from load_data import load_diabetes_data
from preprocessor import Preprocessor
from utils import evaluate

task = 'regression'

def run_single():
    columns, (x_raw_train, y_raw_train), (x_raw_val, y_raw_val), (x_raw_test, y_raw_test) = load_diabetes_data()
    feature_columns = [DenseFeat(col, 1) for col in columns]

    batch_size = int(np.ceil(x_raw_train.shape[0] / 100))

    preprocessor = Preprocessor(task=task, n_quantiles=x_raw_train.shape[0])
    x_train, y_train = preprocessor.fit_transform(x_raw_train, y_raw_train)
    x_val, y_val = preprocessor.transform(x_raw_val, y_raw_val)
    x_test, y_test = preprocessor.transform(x_raw_test, y_raw_test)

    train_model_input = dict(zip(columns, x_train.T))
    val_model_input = dict(zip(columns, x_val.T))
    test_model_input = dict(zip(columns, x_test.T))

    dcn = DCN(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns, dnn_hidden_units=[16, 32, 16], dnn_dropout=0.5, l2_reg_linear=0.0001, l2_reg_embedding=0.0001, l2_reg_cross=0.0001, l2_reg_dnn=0.2, task='regression', device='cpu')
    dcn.compile("adam", "mse", metrics=['mse'])
    dcn.fit(train_model_input, y_train, validation_data=(val_model_input, y_val), epochs=1000)

    pred = dcn.predict(test_model_input)
    r_sq, r_abs = evaluate(y_test, pred, task)
    return r_sq, r_abs


def run_multiple(n=5):
    r_sq_arr = []
    r_abs_arr = []
    for i in range(n):
        r_sq, r_abs = run_single()
        print("run completed...")
        r_sq_arr.append(r_sq)
        r_abs_arr.append(r_abs)
    r_sq_mean = np.mean(r_sq_arr)
    r_sq_std_dev = np.std(r_sq_arr)

    r_abs_mean = np.mean(r_abs_arr)
    r_abs_std = np.std(r_abs_arr)
    return (r_sq_mean, r_sq_std_dev), (r_abs_mean, r_abs_std)

if __name__ == '__main__':
    (r_sq_mean, r_sq_std_dev), (r_abs_mean, r_abs_std) = run_multiple()
    print(f'R_SQ: {r_sq_mean:.7f} (+/- {r_sq_std_dev:.7f})')
    print(f'R_ABS: {r_abs_mean:.7f} (+/- {r_abs_std:.7f})')