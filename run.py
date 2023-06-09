import argparse
import torch
import numpy as np

from collections import namedtuple

from load_data import load_california_housing, load_pima_indians_diabetes_database, load_diabetes_data
from preprocessor import Preprocessor
from model import HONAM
from utils import set_global_seed, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="set the random seed for reproducibility")
parser.add_argument("--mode", choices=["train", "test"], type=str, required=False, help="choose the run mode")
parser.add_argument("--dataset", choices=["california_housing", "pima_indians_diabetes_database", "diabetes_data"], type=str, required=True, help="dataset name")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_info = namedtuple("dataset_info", ["load", "out_size", "task"])
dataset_map = {
    "california_housing": dataset_info(load_california_housing, 1, "regression"),
    "diabetes_data": dataset_info(load_diabetes_data, 1, "regression"),
    "pima_indians_diabetes_database": dataset_info(load_pima_indians_diabetes_database, 1, "binary_classification")
}

def main():

    if args.seed is not None:
        set_global_seed(args.seed)

    # load dataset
    dataset = dataset_map[args.dataset]
    columns, (x_train, y_train), (x_val, y_val), (x_test, y_test) = dataset.load()

    # preprocess dataset
    preprocessor = Preprocessor(task=dataset.task)
    x_train, y_train = preprocessor.fit_transform(x_train, y_train)
    x_val, y_val = preprocessor.transform(x_val, y_val)
    x_test, y_test = preprocessor.transform(x_test, y_test)

    # define model
    num_features = x_train.shape[1]
    model = HONAM(num_features=num_features, out_size=dataset.out_size, task=dataset.task, order=2).to(device)
    model.fit(x_train, y_train, x_val, y_val)
    prediction = model.predict(x_test)

    evaluate(y_test, prediction, task=dataset.task)

    model.forward(torch.tensor(x_test[0].reshape(1, -1), dtype=torch.float32))


if __name__ == '__main__':
    main()
