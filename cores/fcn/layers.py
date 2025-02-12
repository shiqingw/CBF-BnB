import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'sigmoid':
        return torch.sigmoid
    elif activation == 'tanh':
        return torch.tanh
    elif activation == 'softplus':
        return F.softplus
    elif activation == 'identity':
        return lambda x: x
    else:
        raise ValueError("Unsupported activation function: {}".format(activation))

def get_activation_der(activation_name):
    if activation_name == 'relu':
        def relu_derivative(x):
            return (x > 0).to(x.dtype)
        return relu_derivative
    
    elif activation_name == 'sigmoid':
        def sigmoid_derivative(x):
            sig = torch.sigmoid(x)
            return sig * (1 - sig)
        return sigmoid_derivative

    elif activation_name == 'tanh':
        def tanh_derivative(x):
            return 1 - torch.tanh(x) ** 2
        return tanh_derivative
    
    elif activation_name == 'softplus':
        def softplus_derivative(x):
            return torch.sigmoid(x)
        return softplus_derivative

    elif activation_name == 'identity':
        def identity_derivative(x):
            return torch.ones_like(x)
        return identity_derivative
    
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")
    
        
class LinearLayer(nn.Linear): 
    def __init__(self, in_features, out_features, bias=True, activation='relu', dtype=torch.float32):
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         dtype=dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.activation_name = activation
        self.activation = get_activation_fn(activation)
        self.activation_der = get_activation_der(activation)
        
    def forward(self, x):
        if x.dim() != 2:
            raise ValueError("Input tensor must be 2D")
        
        out = self.activation(super().forward(x))
        return out  # Remove the second activation
    
    def forward_with_jacobian(self, x):
        if x.dim() != 2:
            raise ValueError("Input tensor must be 2D")
        
        pre_activation = super().forward(x)
        out = self.activation(pre_activation)
        jac = self.activation_der(pre_activation).unsqueeze(2) * self.weight.unsqueeze(0)
        return out, jac
