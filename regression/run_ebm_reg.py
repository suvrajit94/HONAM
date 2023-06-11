import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor

from load_data import load_diabetes_data
from preprocessor import Preprocessor
from utils import evaluate

task = 'regression'


def run_single():
    _, (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_diabetes_data()

    preprocessor = Preprocessor(task=task, n_quantiles=x_train.shape[0])
    x_train, y_train = preprocessor.fit_transform(x_train, y_train)
    x_val, y_val = preprocessor.transform(x_val, y_val)
    x_test, y_test = preprocessor.transform(x_test, y_test)
    x_train = np.concatenate((x_train, x_val))
    y_train = np.concatenate((y_train, y_val)).ravel()
    # train
    model = ExplainableBoostingRegressor(outer_bags=8, inner_bags=8, learning_rate=0.01, max_rounds=30_000)
    model.fit(x_train, y_train)

    r_sq, r_abs = evaluate(y_test, model.predict(x_test), task)
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