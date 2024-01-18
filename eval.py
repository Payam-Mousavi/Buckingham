import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from predict import predict_model

def eval_model(dataset, model_path):
    ground_truths = [target for _, target in dataset]
    predictions = predict_model(model_path=model_path, dataset=dataset)

    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    mse = mean_squared_error(ground_truths, predictions)
    mae = mean_absolute_error(ground_truths, predictions)

    return mse, mae
