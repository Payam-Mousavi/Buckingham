
import torch
import numpy as np
from torch.utils.data import DataLoader


def predict_model(model_path, dataset, return_intermediate=False):
    model = torch.load(model_path)
    model.eval()

    test_inputs = []
    for i in range(len(dataset)):
        input_data, _ = dataset[i]
        if not torch.is_tensor(input_data):
            input_data = torch.tensor(input_data, dtype=torch.float32)
        test_inputs.append(input_data)

    test_inputs = torch.stack(test_inputs)
    device = next(model.parameters()).device
    test_inputs = test_inputs.to(device)

    if return_intermediate:
        intermediate_outputs = []  # To store intermediate outputs
        predictions = []
        with torch.no_grad():
            for input_data in test_inputs:  # Process each input individually
                output, intermediate = model(input_data.unsqueeze(0), return_intermediate=True)
                # Ensure compatibility with different output shapes
                predictions.append(output.cpu().numpy().squeeze())
                intermediate_outputs.append(intermediate.cpu().numpy().squeeze())
        return predictions, intermediate_outputs
    else:
        with torch.no_grad():
            outputs = model(test_inputs)
            predictions = outputs.cpu().numpy().squeeze()  # Compatibility with different output shapes
        return predictions

# def predict_model(model_path, dataset, return_intermediate=False):
#     model = torch.load(model_path)
#     model.eval()

#     test_inputs = []
#     for i in range(len(dataset)):
#         input_data, _ = dataset[i]
#         if not torch.is_tensor(input_data):
#             input_data = torch.tensor(input_data, dtype=torch.float32)
#         test_inputs.append(input_data)

#     test_inputs = torch.stack(test_inputs)

#     # Ensure the inputs are on the same device as the model
#     device = next(model.parameters()).device
#     test_inputs = test_inputs.to(device)

#     predictions = []

#     if return_intermediate:
#         intermediate_outputs = []  # To store intermediate outputs
#         with torch.no_grad():
#             for input_data in test_inputs:  # Iterate over each input for individual processing
#                 output, intermediate = model(input_data.unsqueeze(0), return_intermediate=True)  # Adjust for individual inputs
#                 predictions.append(output.cpu().numpy().squeeze())
#                 intermediate_outputs.append(intermediate.cpu().numpy().squeeze())
#         return predictions, intermediate_outputs
#     else:
#         with torch.no_grad():
#             outputs = model(test_inputs)
#             predictions = outputs.cpu().numpy().squeeze()  # Ensure it's on CPU for numpy conversion
#         return predictions
    


