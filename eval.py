import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from predict import predict_model

def eval_model(dataset, model_path, visualize=False):
    # Extract Re values from dataset:
    Re_vec = []
    for re, _ in dataset:
        Re_vec.append(re.item())
    Re_vec = np.array(Re_vec)
    # Re_vec = dataset.Re #TODO: check if this works

    ground_truths = [target for _, target in dataset]
    predictions = predict_model(model_path=model_path, dataset=dataset)

    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    mse = mean_squared_error(ground_truths, predictions)
    mae = mean_absolute_error(ground_truths, predictions)

    model_name = model_path.split('/')[-1].split('.')[0]
    figure_save_path = f'Figures/{model_name}_CD_Eval.png'

    if visualize:
        plt.scatter(Re_vec, predictions, color='red', label='CD Predicted', marker='x', alpha=0.3)
        plt.scatter(Re_vec, ground_truths, color='blue', label='CD True', marker='o', alpha=0.3)
        # plt.scatter(Re_vec, np.abs(predictions - ground_truths), color='green', label='CD Error', marker='.', alpha=0.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Reynolds Number (/1E6)')
        plt.ylabel('Drag Coefficient (CD)')
        plt.title('Comparison of True and Predicted CD values')
        plt.legend()

        plt.savefig(figure_save_path, dpi=300, bbox_inches='tight')
        plt.show()

    return mse, mae
