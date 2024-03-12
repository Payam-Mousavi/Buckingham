
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
        x = F.softplus(x) #TODO: check this?
        return x

# Model_I
class Model_I(nn.Module):
    def __init__(self, input_dim=4, output_dim=1, hidden_dims=[32, 16, 8, 4, 1, 64, 32, 16, 8, 4]):
        super(Model_I, self).__init__()
        hidden_dims.insert(0, input_dim)  # Input dimension at the beginning
        hidden_dims.append(output_dim)  # Output dimension at the end
        self.layers = nn.ModuleList()
        self.output_sizes = hidden_dims[1:]  # Adjusted to track output sizes directly from hidden_dims
        for i in range(len(hidden_dims) - 1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
            self.layers.append(layer)

    def forward(self, x, return_intermediate=False):
        intermediate_output = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply Leaky ReLU for all but last layer
                x = F.leaky_relu(x, 0.1)
                if return_intermediate and self.output_sizes[i] == 1:
                    # Update intermediate_output only for layers before the final layer
                    intermediate_output = x
            else:
                # For the last layer, apply softplus without updating intermediate_output
                x = F.softplus(x)

        if return_intermediate and intermediate_output is not None:
            return x, intermediate_output
        else:
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
            # x = torch.tanh(self.layers[i](x))
        x = self.layers[-1](x)  # Apply the last layer without ReLU
        x = F.softplus(x)
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