import numpy as np
import random
import torch

from sklearn.metrics import roc_auc_score, average_precision_score
from torch.backends import cudnn


def set_global_seed(seed: int) -> None:

    """
    Set the global random seed for reproducibility.
    :param seed: a random seed number
    """

    cudnn.deterministic = True
    cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_r_squared(y_true: np.ndarray, y_predict: np.ndarray):
    mean = np.mean(y_true)
    sst = np.square(y_true - mean).sum()
    ssr = np.square(y_true - y_predict).sum()
    score = 1 - (ssr / sst)
    return score


def get_r_absolute(y_true: np.ndarray, y_predict: np.ndarray):
    mean = np.mean(y_true)
    sat = np.abs(y_true - mean).sum()
    sar = np.abs(y_true - y_predict).sum()
    score = 1 - (sar / sat)
    return score


def evaluate(y_true: np.ndarray, y_predict: np.ndarray, task: str):
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_predict.ndim == 1:
        y_predict = y_predict.reshape((-1, 1))
    if task == "regression":
        r_squared = get_r_squared(y_true, y_predict)
        r_absolute = get_r_absolute(y_true, y_predict)
        return r_squared, r_absolute
    elif task in {"binary_classification", "classification"}:
        auroc = roc_auc_score(y_true, y_predict)
        auprc = average_precision_score(y_true, y_predict)
        return auroc, auprc
