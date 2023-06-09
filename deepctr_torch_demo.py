import torch
from deepctr_torch.inputs import DenseFeat
from deepctr_torch.models import DCN

from load_data import load_diabetes_data
from preprocessor import Preprocessor
from utils import evaluate

if __name__ == "__main__":
    task = 'regression'
    columns, (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_diabetes_data()

    # preprocess dataset
    preprocessor = Preprocessor(task=task)
    x_train, y_train = preprocessor.fit_transform(x_train, y_train)
    x_val, y_val = preprocessor.transform(x_val, y_val)
    x_test, y_test = preprocessor.transform(x_test, y_test)

    feature_columns = [DenseFeat(col, 1) for col in columns]

    # 3.generate input data for model
    train_model_input = dict(zip(columns, x_train.T))
    val_model_input = dict(zip(columns, x_val.T))
    test_model_input = dict(zip(columns, x_test.T))

    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DCN(feature_columns, feature_columns, task=task, device=device)
    model.compile("adam", "mse", metrics=['mse'], )

    history = model.fit(train_model_input, y_train, validation_data=(val_model_input, y_val), epochs=1000)
    pred_ans = model.predict(test_model_input)

    evaluate(y_test, pred_ans, task)