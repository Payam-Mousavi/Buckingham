
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Model_II
class Model_II(nn.Module):
    def __init__(self, input_dim=4 , output_dim= 1, hidden_dims=[32, 16, 8, 4]):
        super(Model_II, self).__init__()
        hidden_dims.insert(0, input_dim)
        hidden_dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
            self.layers.append(layer)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            # x = torch.relu(self.layers[i](x))
            x = F.leaky_relu(self.layers[i](x), 0.1)
            # x = torch.tanh(self.layers[i](x))
        x = self.layers[-1](x)  # Apply the last layer without ReLU
        return x
    
class Model_0(nn.Module):
    def __init__(self, input_dim=1 , output_dim= 1, hidden_dims=[32, 16, 8, 4]):
        super(Model_0, self).__init__()
        assert input_dim == 1, "input_dim must be 1 for Model_0"
        hidden_dims.insert(0, input_dim)
        hidden_dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
            self.layers.append(layer)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.leaky_relu(self.layers[i](x), 0.1)
        x = self.layers[-1](x)  # Apply the last layer without ReLU
        return x

# Model_II with BatchNorm
# class Model_II(nn.Module):
#     def __init__(self, input_dim=4, output_dim=1, hidden_dims=[32, 16, 8, 4]):
#         super(Model_II, self).__init__()
#         hidden_dims.insert(0, input_dim)
#         hidden_dims.append(output_dim)
#         self.layers = nn.ModuleList()
#         self.batch_norms = nn.ModuleList()

#         for i in range(len(hidden_dims) - 1):
#             self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
#             # Add BatchNorm layer for each hidden layer, but not for the output layer
#             if i < len(hidden_dims) - 2:
#                 self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i+1]))

#     def forward(self, x):
#         for i in range(len(self.layers) - 1):
#             x = self.layers[i](x)
#             x = self.batch_norms[i](x)
#             x = F.leaky_relu(x, 0.1)

#         x = self.layers[-1](x)  # Apply the last layer without ReLU
#         return x