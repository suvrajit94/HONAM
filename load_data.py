from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def _split_data(
        df: pd.DataFrame, test_size: float, target_column: str
) -> [List[str], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load a dataset from the given path and split it into train/val/test sets.
    :param df: dataframe
    :param test_size: test set ratio
    :param target_column: dataset target column name
    :return: train/val/test sets if is_val is true else train/test
    """
    # split the dataframe into train/val/test sets
    assert 0. < test_size < 1.
    test_size = int(test_size * len(df))
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)  # the test set should be the same
    df_train, df_val = train_test_split(df_train, test_size=test_size)

    # divide into feature and target arrays
    y_train, x_train = df_train.pop(target_column).values.reshape(-1, 1), df_train.values
    y_val, x_val = df_val.pop(target_column).values.reshape(-1, 1), df_val.values
    y_test, x_test = df_test.pop(target_column).values.reshape(-1, 1), df_test.values

    return list(df_train.columns), (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_california_housing(
        path: str = "./data/california_housing.csv",
        test_size: float = 0.2
) -> Tuple[List[str], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load the California Housing dataset.
    :param path: dataset path
    :param test_size: test set ratio
    :return: train/val/test sets of the California Housing dataset
    """
    df = pd.read_csv(path).dropna(axis=1)
    return _split_data(df, test_size, "median_house_value")


def load_pima_indians_diabetes_database(
        path: str = "./data/pima_indians_diabetes_database.csv",
        test_size: float = 0.2
) -> Tuple[List[str], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    :param path: dataset path
    :param test_size: test set ratio
    :return: train/val/test sets of the Pima Indians Diabetes Database
    """
    df = pd.read_csv(path).dropna(axis=1)
    imputer = SimpleImputer(missing_values=0, strategy='mean')
    columns_to_impute = [col for col in df.columns if col not in {'Outcome', 'Pregnancies'}]
    columns_to_impute = df[columns_to_impute].applymap(lambda x: x == 0).sum()
    columns_to_impute = [index for (index, value) in columns_to_impute.items() if value > 0]
    for col in columns_to_impute:
        df[col] = imputer.fit_transform(df[[col]]).ravel()
    return _split_data(df, test_size, target_column="Outcome")


def load_diabetes_data(
        path: str = "./data/diabetes_data.csv",
        test_size: float = 0.2
) -> Tuple[List[str], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
    :param path: dataset path
    :param test_size: test set ratio
    :return: train/val/test sets of the Diabetes Data
    """
    df = pd.read_csv(path).dropna(axis=1)
    return _split_data(df, test_size, target_column="y")


if __name__ == '__main__':
    load_diabetes_data()
