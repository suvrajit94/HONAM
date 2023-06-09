import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from torch import nn


class ClassifierModule(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            dropout=0.5,
    ):
        super(ClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X


if __name__ == '__main__':
    mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
    X = mnist.data.astype('float32')
    X /= 255.0

    y = mnist.target.astype('int64')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mnist_dim = X.shape[1]
    hidden_dim = int(mnist_dim / 8)
    output_dim = len(np.unique(mnist.target))

    torch.manual_seed(0)

    net = NeuralNetClassifier(
        ClassifierModule,
        max_epochs=20,
        lr=0.1,
        device=device,
        module__input_dim=mnist_dim,
        module__hidden_dim=hidden_dim,
        module__output_dim=output_dim
    )

    net.fit(X_train, y_train)

    y_pred = net.predict(X_test)

    print(accuracy_score(y_test, y_pred))
