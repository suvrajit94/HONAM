import numpy as np
from sklearn.linear_model import LinearRegression

from load_data import load_diabetes_data
from preprocessor import Preprocessor
from utils import evaluate

task = 'regression'

def run_single():
    columns, (x_raw_train, y_raw_train), (x_raw_val, y_raw_val), (x_raw_test, y_raw_test) = load_diabetes_data()
    lr = LinearRegression()
    preprocessor = Preprocessor(task=task, n_quantiles=x_raw_train.shape[0])
    x_train, y_train = preprocessor.fit_transform(x_raw_train, y_raw_train)
    x_val, y_val = preprocessor.transform(x_raw_val, y_raw_val)
    x_test, y_test = preprocessor.transform(x_raw_test, y_raw_test)

    lr.fit(x_train, y_train)
    lr_fit = lr.predict(x_train)
    lr_pred = lr.predict(x_test)
    r_sq, r_abs = evaluate(y_test, lr_pred, task)
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