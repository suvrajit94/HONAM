import numpy as np

from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from typing import Tuple, List


class Preprocessor:

    def __init__(self, task: str, n_quantiles: int = 1000):
        """
        :param task: regression or binary_classification
        :param n_quantiles: the number of quantiles
        """

        self._task = task

        self._ordinary_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self._quantile_transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal")
        self._standard_scaler = StandardScaler()

        if task == "regression":
            self._y_transformer = StandardScaler()
        elif task == "binary_classification":
            self._y_transformer = OrdinalEncoder()

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:

        """
        Fit preprocessor.
        :param x: feature
        :param y: target
        """

        self._transform_xy(x, y, fit=True)

    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """
        Transform x and y.
        :param x: feature
        :param y: target
        :return: transformed x and y
        """

        x, y = self._transform_xy(x, y)

        return x, y

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """
        Fit preprocessor then transform x and y.
        :param x: feature
        :param y: target
        :return: transformed x and y
        """

        x, y = self._transform_xy(x, y, fit=True)

        return x, y

    def inverse_transform_x(self, x: np.ndarray) -> np.ndarray:
        """
        Inverse transform x
        :param x: transformed feature
        :return: original x
        """
        _x = self._standard_scaler.inverse_transform(x)
        _x = self._quantile_transformer.inverse_transform(_x)
        if self._cat_idx:
            _x[:, self._cat_idx] = self._ordinary_encoder.inverse_transform(_x[:, self._cat_idx])
        return _x

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform y.
        :param y: transformed target
        :return: original y
        """
        return self._y_transformer.inverse_transform(y)

    def _transform_xy(self, x: np.ndarray, y: np.ndarray, fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:

        self._cat_idx = self._get_cat_idx(x)

        _x, _y = x.copy(), y.copy()
        if fit:
            if self._cat_idx:
                _x[:, self._cat_idx] = self._ordinary_encoder.fit_transform(_x[:, self._cat_idx])
            _x = self._quantile_transformer.fit_transform(_x)
            _x = self._standard_scaler.fit_transform(_x)
            _y = self._y_transformer.fit_transform(_y)
        else:
            if self._cat_idx:
                _x[:, self._cat_idx] = self._ordinary_encoder.transform(_x[:, self._cat_idx])
            _x = self._quantile_transformer.transform(_x)
            _x = self._standard_scaler.transform(_x)
            _y = self._y_transformer.transform(_y)

        return _x, _y

    @ staticmethod
    def _get_cat_idx(x: np.ndarray) -> List[int]:

        """
        Get categorical features' indices.
        :param x: feature
        :return: categorical features' indices
        """

        cat_idx = []
        num_features = x.shape[1]
        for i in range(num_features):
            try:
                x[:, i].astype(float)
            except ValueError:
                cat_idx.append(i)

        return cat_idx
