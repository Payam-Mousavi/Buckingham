
import torch
import numpy as np
from torch.utils.data import DataLoader


# def predict_model(model_path, dataset):
#     # Load the entire model
#     model = torch.load(model_path)
#     model.eval()

#     test_inputs = []
#     for i in range(len(dataset)):
#         input, _ = dataset[i]
#         test_inputs.append(input)

#     test_inputs = torch.tensor(np.stack(test_inputs), dtype=torch.float32)

#     with torch.no_grad():
#         predictions = model(test_inputs).squeeze().numpy()
        

#     return predictions


def predict_model(model_path, dataset):
    model = torch.load(model_path)
    model.eval()

    test_inputs = []
    for i in range(len(dataset)):
        input_data, _ = dataset[i]
        if not torch.is_tensor(input_data):
            input_data = torch.tensor(input_data, dtype=torch.float32)
        test_inputs.append(input_data)

    test_inputs = torch.stack(test_inputs)
    
    # Ensure the inputs are on the same device as the model
    device = next(model.parameters()).device
    test_inputs = test_inputs.to(device)

    with torch.no_grad():
        outputs = model(test_inputs)
        predictions = outputs.cpu().numpy().squeeze()  # Ensure it's on CPU for numpy conversion

    return predictions

