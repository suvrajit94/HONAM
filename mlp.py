import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from skorch import NeuralNetClassifier, NeuralNetRegressor


from load_data import load_diabetes_data, load_pima_indians_diabetes_database
from preprocessor import Preprocessor
from utils import evaluate


class MLP(nn.Module):

    def __init__(self,
                 num_features: int,
                 out_size: int,
                 hidden_layers: tuple[int] = (32, 64, 32),
                 dropout: float = None,
                 task='regression'):
        super(MLP, self).__init__()

        self._model = nn.Sequential()
        # input layer
        if dropout:
            self._model.add_module("dropout_input", nn.Dropout(dropout))
        self._model.add_module("input", nn.Linear(num_features, hidden_layers[0]))
        self._model.add_module("leaky_relu_input", nn.LeakyReLU())

        # hidden layers
        for i, in_features in enumerate(hidden_layers[:-1]):
            if dropout:
                self._model.add_module(f"dropout_{i}", nn.Dropout(dropout))
            self._model.add_module(f"linear_{i}", nn.Linear(in_features, hidden_layers[i+1]))
            self._model.add_module(f"leaky_relu_{i}", nn.LeakyReLU())

        # output layer
        if dropout:
            self._model.add_module(f"dropout_output", nn.Dropout(dropout))
        self._model.add_module(f"output", nn.Linear(hidden_layers[-1], out_size))

        if task == 'classification':
            self.model.add_module('output', nn.Softmax().Sigmoid())
            self._loss_fn = nn.CrossEntropyLoss()
        elif task == 'binary_classification':
            self.model.add_module('sigmoid', nn.Sigmoid())
            self._loss_fn = nn.BCELoss()
        elif task == 'regression':
            self._loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Task {task} is not supported.")

    def forward(self, x):
        return self._model(x)


def get_regressor(num_features: int,
                  out_size: int,
                  hidden_layers: tuple[int] = (32, 64, 32),
                  dropout: float = None,
                  max_epochs: int = 1000,
                  batch_size: int = 32,
                  lr: float = 0.001,
                  weight_decay: float = 0.0001):
    return NeuralNetRegressor(
        module=MLP,
        module__num_features=num_features,
        module__out_size=out_size,
        module__hidden_layers=hidden_layers,
        module__dropout=dropout,
        module__task='regression',
        max_epochs=max_epochs,
        batch_size=batch_size,
        optimizer=optim.Adam,
        lr=lr,
        optimizer__weight_decay=weight_decay
    )


def get_binary_classifier(num_features: int,
                          out_size: int,
                          hidden_layers: tuple[int] = (32, 64, 32),
                          dropout: float = None,
                          max_epochs: int = 1000,
                          batch_size: int = 32,
                          lr: float = 0.001,
                          weight_decay: float = 0.0001):
    return NeuralNetClassifier(
        module=MLP,
        module__num_features=num_features,
        module__out_size=out_size,
        module__hidden_layers=hidden_layers,
        module__dropout=dropout,
        module__task='binary_classification',
        max_epochs=max_epochs,
        batch_size=batch_size,
        optimizer=optim.Adam,
        lr=lr,
        optimizer__weight_decay=weight_decay
    )

    # def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
    #     x_train = torch.tensor(x_train, dtype=torch.float32)
    #     y_train = torch.tensor(y_train, dtype=torch.float32)
    #
    #     x_val = torch.tensor(x_val, dtype=torch.float32)
    #     y_val = torch.tensor(y_val, dtype=torch.float32)
    #
    #     batch_size = self._batch_size if self._batch_size else int(np.ceil(x_train.shape[0] / 100))
    #     batch_start = torch.arange(0, x_train.shape[0], batch_size)
    #
    #     optimizer = optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
    #
    #     best_loss = np.inf   # init to infinity
    #     for epoch in range(self._n_epochs):
    #         with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=not self._verbose) as bar:
    #             bar.set_description(f"Epoch {epoch}")
    #             for start in bar:
    #                 self.model.train()
    #                 # take a batch
    #                 x_batch = x_train[start:start+batch_size]
    #                 y_batch = y_train[start:start+batch_size]
    #                 # forward pass
    #                 y_pred = self.model(x_batch)
    #                 loss = self._loss_fn(y_pred, y_batch)
    #                 # backward pass
    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 # update weights
    #                 optimizer.step()
    #
    #                 # evaluate accuracy at end of each epoch
    #                 self.model.eval()
    #                 y_pred = self.model(x_val)
    #                 val_loss = self._loss_fn(y_pred, y_val)
    #                 val_loss = float(val_loss)
    #                 self.history.append(val_loss)
    #                 if val_loss < best_loss:
    #                     self._best_weights = copy.deepcopy(self.model.state_dict())
    #                 # print progress
    #                 bar.set_postfix(train_mse=float(loss), val_loss=val_loss)
    #
    # def predict(self, x: np.ndarray) -> np.ndarray:
    #     self.model.eval()
    #     self.model.load_state_dict(self._best_weights)
    #     x_test = torch.tensor(x, dtype=torch.float32)
    #     return self.model(x_test).detach().numpy()


def main(task='regression'):
    lr = 0.001
    n_epochs = 1000   # number of epochs to run
    if task == 'regression':
        columns, (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_diabetes_data()
    elif task == 'binary_classification':
        columns, (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_pima_indians_diabetes_database()
    else:
        raise ValueError(f"Task {task} is not supported.")

    preprocessor = Preprocessor(task=task)
    x_train, y_train = preprocessor.fit_transform(x_train, y_train)
    x_val, y_val = preprocessor.transform(x_val, y_val)
    x_test, y_test = preprocessor.transform(x_test, y_test)

    mlp = MLP(x_train.shape[1], 1, batch_size=None, n_epochs=n_epochs, verbose=True, lr=lr, task=task)
    mlp.fit(x_train, y_train, x_val, y_val)

    evaluate(y_train, mlp.predict(x_train), task)
    evaluate(y_val, mlp.predict(x_val), task)
    evaluate(y_test, mlp.predict(x_test), task)


if __name__ == '__main__':
    main(task='binary_classification')
