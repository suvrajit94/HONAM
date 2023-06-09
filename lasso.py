from sklearn import linear_model
from sklearn.metrics import r2_score
from load_data import load_diabetes_data
from preprocessor import Preprocessor
from utils import evaluate


def main():
    task = 'regression'
    columns, (x_train, y_train), (x_test, y_test) = load_diabetes_data(is_val=False)

    preprocessor = Preprocessor(task=task)
    x_train, y_train = preprocessor.fit_transform(x_train, y_train)
    x_test, y_test = preprocessor.transform(x_test, y_test)

    lasso = linear_model.Lasso(alpha=0.0014)
    lasso.fit(x_train, y_train)
    lasso_pred = lasso.predict(x_test)
    print(r2_score(y_test, lasso_pred))
    evaluate(y_test, lasso_pred, task)


if __name__ == '__main__':
    main()
