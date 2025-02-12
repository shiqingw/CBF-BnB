import torch 
import torch.nn as nn 
from .layers import LinearLayer

def get_activation(activation_name):
    if activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'sigmoid':
        return nn.Sigmoid()
    elif activation_name == 'tanh':
        return nn.Tanh()
    elif activation_name == 'softplus':
        return nn.Softplus()
    elif activation_name == 'identity':
        return nn.Identity()
    elif activation_name == 'leaky_relu':
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

def get_activation_first_der_bound(activation_name):
    if activation_name == 'sigmoid':
        return 0.25
    elif activation_name == 'tanh':
        return 1.0
    elif activation_name == 'softplus':
        return 1.0
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

def get_activation_second_der_bound(activation_name):
    if activation_name == 'sigmoid':
        return 0.09623
    elif activation_name == 'tanh':
        return 0.7699
    elif activation_name == 'softplus':
        return 0.25
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

class FullyConnectedNetwork(nn.Module):
    def __init__(self, in_features, out_features, activations, widths, zero_at_zero=False, 
                 input_bias=None, input_transform=None, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activations = activations  # List of activation function names
        self.widths = widths  # List of widths for each layer
        self.zero_at_zero = zero_at_zero
        self.dtype = dtype
        if input_bias is None:
            input_bias = torch.zeros(in_features, dtype=self.dtype)
        else:
            input_bias = torch.tensor(input_bias, dtype=self.dtype)
        if input_transform is None:
            input_transform = torch.ones(in_features, dtype=self.dtype)
        else:
            input_transform = torch.tensor(input_transform, dtype=self.dtype)
        self.register_buffer('input_bias', input_bias)
        self.register_buffer('input_transform', input_transform)
        
        if len(self.activations) != len(self.widths) - 2:
            raise ValueError("Number of activations must be two less than number of widths. The last layer has no activation.")
        if self.widths[-1] != self.out_features:
            raise ValueError("Last width must match number of output channels.")
        if self.widths[0] != self.in_features:
            raise ValueError("First width must match number of input channels.")
        
        layers = []
        for i in range(len(self.activations)):
            layer = LinearLayer(in_features=self.widths[i],
                                out_features=self.widths[i+1],
                                bias=True,
                                activation=self.activations[i],
                                dtype=self.dtype)
            
            layers.append(layer)

        layer = LinearLayer(in_features=self.widths[-2],
                            out_features=self.widths[-1],
                            bias=True,
                            activation='identity',
                            dtype=self.dtype)
        layers.append(layer)
    
        self.model = nn.Sequential(*layers)
        self.layers = layers
            
    def forward(self, x):
        x = (x-self.input_bias) * self.input_transform
        out = self.model(x)

        if self.zero_at_zero:
            zeros = torch.zeros_like(x)
            zeros = (zeros-self.input_bias) * self.input_transform
            zero_values = self.model(zeros)
            out = out - zero_values
            
        return out
    
    def forward_with_jacobian(self, x):
        out = (x-self.input_bias) * self.input_transform
        input_transform_expanded = self.input_transform.expand(x.shape[0], -1)
        J = torch.diag_embed(input_transform_expanded).to(self.dtype)

        for layer in self.layers:
            out, jac = layer.forward_with_jacobian(out)
            J = torch.bmm(jac, J)
        
        if self.zero_at_zero:
            zeros = torch.zeros_like(x)
            zeros = (zeros-self.input_bias) * self.input_transform
            zero_values = self.model(zeros)
            out = out - zero_values

        return out, J
    
    def get_l2_hessian_bound(self, which_output=0):

        assert which_output < self.out_features
        assert len(self.activations) == len(self.layers) - 1
        bound = 0.0
        input_transform_lipschitz = torch.max(torch.abs(self.input_transform)).item()
        L = len(self.layers) # number of layers

        dz_dx_norm = [] # shape: (L,)
        for i in range(L):
            layer = self.layers[i]
            if i == 0:
                r = torch.norm(layer.weight, p=2).item() * input_transform_lipschitz
            else:
                g = get_activation_first_der_bound(self.activations[i-1])
                r = dz_dx_norm[i-1] * torch.norm(layer.weight, p=2).item() * g
            dz_dx_norm.append(r)
        assert len(dz_dx_norm) == L

        da_da_abs = [] # shape: (L-1,) of inhomogenius 2D tensors
        for i in range(L-1):
            layer = self.layers[i]
            g = get_activation_first_der_bound(self.activations[i-1])
            weight_abs = torch.abs(layer.weight) * g # shape: (n_(l+1), n_l)
            da_da_abs.append(weight_abs)
        assert len(da_da_abs) == L-1

        daLminus1_dal_abs = [] # shape: (L-2,) of inhomogenius 2D tensors
        for i in range(L-2):
            if i == 0:
                daLminus1_dal_abs.append(da_da_abs[L-2-i])
            else:
                daLminus1_dal_abs.append(torch.mm(daLminus1_dal_abs[i-1], da_da_abs[L-2-i]))
            assert daLminus1_dal_abs[i].shape[0] == self.layers[L-2].out_features
            assert daLminus1_dal_abs[i].shape[1] == self.layers[L-3-i].out_features
        daLminus1_dal_abs = daLminus1_dal_abs[::-1] # reverse the list
        assert len(daLminus1_dal_abs) == L-2

        dzLi_dal_abs = [] # shape: (L-1,) of inhomogenius 2D tensors
        WL_i_abs = torch.abs(self.layers[-1].weight)[which_output].unsqueeze(0) # shape: (1, n_(L-1))
        for i in range(L-1):
            if i == 0:
                dzLi_dal_abs.append(WL_i_abs)
            else:
                dzLi_dal_abs.append(torch.mm(WL_i_abs, daLminus1_dal_abs[i-1]))
            assert dzLi_dal_abs[i].shape[0] == 1
            assert dzLi_dal_abs[i].shape[1] == self.layers[i].out_features
        assert len(dzLi_dal_abs) == L-1

        for i in range(L-1):
            h = get_activation_second_der_bound(self.activations[i])
            bound += h * dz_dx_norm[i]**2 * torch.max(dzLi_dal_abs[i]).item()
            
        return bound 

