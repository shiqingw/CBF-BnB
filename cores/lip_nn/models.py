import torch 
import torch.nn as nn 
from .layers import SandwichFc, SandwichLin, cayley
import math

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

def get_activation_second_der_bound(activation_name):
    if activation_name == 'sigmoid':
        return 0.09623
    elif activation_name == 'tanh':
        return 0.7699
    elif activation_name == 'softplus':
        return 0.25
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

def get_activation_third_der_bound(activation_name):
    if activation_name == 'sigmoid':
        return 1.0/8.0
    elif activation_name == 'tanh':
        return 2.0
    elif activation_name == 'softplus':
        return 0.09623
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

class LipschitzNetwork(nn.Module):
    def __init__(self, in_features, out_features, gamma, activations, widths, zero_at_zero=False, 
                 input_bias=None, input_transform=None, dtype=torch.float32, random_psi=False, trainable_psi=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma = gamma 
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
            if i == 0:
                scale = math.sqrt(self.gamma)
            else:
                scale = 1.0
            layers.append(SandwichFc(in_features=self.widths[i],
                                     out_features=self.widths[i + 1], 
                                     bias=True, 
                                     activation=self.activations[i], 
                                     scale=scale,
                                     dtype=self.dtype, 
                                     random_psi=random_psi,
                                     trainable_psi=trainable_psi))
        layers.append(SandwichLin(in_features=self.widths[-2], 
                                  out_features=self.out_features, 
                                  bias=True, 
                                  scale=math.sqrt(self.gamma), 
                                  AB=False,
                                  dtype=self.dtype))  # Last layer with identity activation
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
    
    def forward_with_jacobian_and_hessian(self, x):
        out = (x-self.input_bias) * self.input_transform
        input_transform_expanded = self.input_transform.expand(x.shape[0], -1)
        J = torch.diag_embed(input_transform_expanded).to(self.dtype)
        H = torch.zeros(x.shape[0], self.in_features, self.in_features, self.in_features, dtype=self.dtype, device=x.device)

        for layer in self.layers:
            out, jac, hess = layer.forward_with_jacobian_and_hessian(out)
            H_new_part_1 = torch.einsum(
                "bjm,bijk,bkn->bimn",
                J,     # (N, hidden_{l-1}, n_x)
                hess,  # (N, hidden_{l}, hidden_{l-1}, hidden_{l-1})
                J,     # (N, hidden_{l-1}, n_x)
                ) # (N, hidden_{l}, n_x, n_x)
            H_new_part_2 = torch.einsum(
                "bij,bjmn->bimn",
                jac,     # (N, hidden_{l}, hidden_{l-1})
                H,   # (N, hidden_{l-1}, n_x, n_x)
                ) # (N, hidden_{l}, n_x, n_x)
            H = H_new_part_1 + H_new_part_2
            J = torch.bmm(jac, J)
        
        if self.zero_at_zero:
            zeros = torch.zeros_like(x)
            zeros = (zeros-self.input_bias) * self.input_transform
            zero_values = self.model(zeros)
            out = out - zero_values

        return out, J, H
    
    def forward_with_jacobian_and_hessian_method2(self, x):
        out = (x-self.input_bias) * self.input_transform
        input_transform_expanded = self.input_transform.expand(x.shape[0], -1)
        J = torch.diag_embed(input_transform_expanded).to(self.dtype)

        leg_list = []
        local_jac_list = []
        W2_list = []
        sigma_second_der_list = []

        for layer in self.layers:
            out, jac, W1, W2, sigma_second_der = layer.forward_with_jacobian_and_hessian_method2(out)
            leg_list.append(torch.matmul(W1, J))
            local_jac_list.append(jac)
            W2_list.append(W2)
            sigma_second_der_list.append(sigma_second_der)
            J = torch.bmm(jac, J)

        backward_jac = torch.eye(self.out_features).unsqueeze(0).repeat(x.shape[0], 1, 1).to(x.device) # (N, n_out, n_out)
        H = torch.zeros(x.shape[0], self.out_features, self.in_features, self.in_features, dtype=self.dtype, device=x.device)

        for i in range(len(self.layers)-2, -1, -1):
            leg = leg_list[i] # (N, hidden_{l}, n_x)
            sigma_second_der = sigma_second_der_list[i] # (N, hidden_{l})
            local_jac = local_jac_list[i+1] # (N, n_out, n_{l})
            W2 = W2_list[i] # (n_{l}, hidden_{l})

            backward_jac = torch.bmm(backward_jac, local_jac) # (N, n_out, n_{l})
            H_new_part = torch.einsum(
                "bji,bj,bkj,bjm->bkim",
                leg, # (N, hidden_{l}, n_x)
                sigma_second_der, # (N, hidden_{l})
                torch.matmul(backward_jac, W2), # (N, n_out, hidden_{l})
                leg # (N, hidden_{l}, n_x)
                )
            
            H += H_new_part

        if self.zero_at_zero:
            zeros = torch.zeros_like(x)
            zeros = (zeros-self.input_bias) * self.input_transform
            zero_values = self.model(zeros)
            out = out - zero_values

        return out, J, H
    
    def get_l2_lipschitz_bound(self):
        input_transform_lipschitz = torch.max(torch.abs(self.input_transform)).item()
        return self.gamma * input_transform_lipschitz
    
    def get_l2_hessian_bound(self):
        bound = 0.0
        input_transform_lipschitz = torch.max(torch.abs(self.input_transform)).item()
        
        assert len(self.activations) == len(self.layers) - 1
        for (i, layer) in enumerate(self.layers):
            if isinstance(layer, SandwichFc):
                activation_bound = get_activation_second_der_bound(self.activations[i])
                psi = layer.psi.detach()
                min_psi = torch.min(psi).item()
                max_psi = torch.max(psi).item()
                bound += (2.0*self.gamma)**(3.0/2) * activation_bound * math.exp(-2*min_psi) * math.exp(max_psi) * input_transform_lipschitz**2

        return bound 
    
    def get_l2_elementwise_third_order_bound(self):

        gamma = self.gamma
        
        activation_bound_second_order_list = []
        activation_bound_third_order_list = []
        min_psi_list = []
        max_psi_list = []
        h_k_h_k_minus_one_dx_j_norm = []
        h_ell_dx_dx_j_norm_list = []

        assert len(self.activations) == len(self.layers) - 1
        for (i, layer) in enumerate(self.layers):
            if isinstance(layer, SandwichFc):
                activation_bound_second_order = get_activation_second_der_bound(self.activations[i])
                activation_bound_third_order = get_activation_third_der_bound(self.activations[i])
                psi = layer.psi.detach()
                min_psi = torch.min(psi).item()
                max_psi = torch.max(psi).item()
                
                h_dx_dx = 2.0**1.5 * gamma * activation_bound_second_order * math.exp(-2*min_psi) * math.exp(max_psi)
                if i > 0:
                    h_dx_dx += h_ell_dx_dx_j_norm_list[i-1]

                h_k_h_k_minus_one_dx_j = 2.0**1.5 * gamma**0.5 * activation_bound_second_order * math.exp(-2*min_psi) * math.exp(max_psi)

                activation_bound_second_order_list.append(activation_bound_second_order)
                activation_bound_third_order_list.append(activation_bound_third_order)
                min_psi_list.append(min_psi)
                max_psi_list.append(max_psi)
                h_ell_dx_dx_j_norm_list.append(h_dx_dx)
                h_k_h_k_minus_one_dx_j_norm.append(h_k_h_k_minus_one_dx_j)

        h_L_dh_ell_dx_j_norm_list = [0.0]
        tmp = 0.0
        for i in range(len(h_k_h_k_minus_one_dx_j_norm)-1, 0, -1):
            tmp += gamma**0.5 * h_k_h_k_minus_one_dx_j_norm[i-1]
            h_L_dh_ell_dx_j_norm_list.append(tmp)
        h_L_dh_ell_dx_j_norm_list.reverse()

        input_transform_lipschitz = torch.max(torch.abs(self.input_transform)).item()
        bound = 0.0

        for (i, layer) in enumerate(self.layers):
            if isinstance(layer, SandwichFc):
                activation_bound_second_order = activation_bound_second_order_list[i]
                activation_bound_third_order = activation_bound_third_order_list[i]
                min_psi = min_psi_list[i]
                max_psi = max_psi_list[i]

                if i > 0:
                    bound += 2**2.5 * gamma * math.exp(-2*min_psi) * math.exp(max_psi) * activation_bound_second_order * h_ell_dx_dx_j_norm_list[i-1]
                
                bound += 4 * gamma**2 * math.exp(-3*min_psi) * math.exp(max_psi) * activation_bound_third_order

                bound += 2.0**1.5 * gamma * math.exp(-2*min_psi) * math.exp(max_psi) * activation_bound_second_order * h_L_dh_ell_dx_j_norm_list[i]

        bound = bound * input_transform_lipschitz**3
        return bound

class ControllerNetwork(nn.Module):
    def __init__(self, in_features, out_features, gamma, activations, widths, zero_at_zero=False, 
                 input_bias=None, input_transform=None, lower_bound=None, upper_bound=None, dtype=torch.float32,
                 random_psi=False, trainable_psi=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma = gamma 
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

        assert input_transform.shape[0] == in_features
        assert input_bias.shape[0] == in_features

        if lower_bound is not None:
            self.register_buffer('lower_bound', torch.tensor(lower_bound, dtype=self.dtype))
            assert self.lower_bound.shape[0] == out_features
        else:
            self.lower_bound = None

        if upper_bound is not None:
            self.register_buffer('upper_bound', torch.tensor(upper_bound, dtype=self.dtype))
            assert self.lower_bound.shape[0] == out_features
        else:
            self.upper_bound = None
        
        if len(self.activations) != len(self.widths) - 2:
            raise ValueError("Number of activations must be two less than number of widths. The last layer has no activation.")
        if self.widths[-1] != self.out_features:
            raise ValueError("Last width must match number of output channels.")
        if self.widths[0] != self.in_features:
            raise ValueError("First width must match number of input channels.")
        
        layers = []
        for i in range(len(self.activations)):
            if i == 0:
                scale = math.sqrt(self.gamma)
            else:
                scale = 1.0
            layers.append(SandwichFc(in_features=self.widths[i],
                                     out_features=self.widths[i + 1], 
                                     bias=True, 
                                     activation=self.activations[i], 
                                     scale=scale,
                                     dtype=self.dtype,
                                     random_psi=random_psi,
                                     trainable_psi=trainable_psi))
        layers.append(SandwichLin(in_features=self.widths[-2], 
                                  out_features=self.out_features, 
                                  bias=True, 
                                  scale=math.sqrt(self.gamma), 
                                  AB=False,
                                  dtype=self.dtype))  # Last layer with identity activation
        self.model = nn.Sequential(*layers)
        self.layers = layers
    
    def forward(self, x):
        x = (x - self.input_bias) * self.input_transform
        out = self.model(x)

        if self.zero_at_zero:
            zeros = torch.zeros_like(x)
            zeros = (zeros - self.input_bias) * self.input_transform
            zero_values = self.model(zeros)
            zero_values = torch.clamp(zero_values, min=self.lower_bound, max=self.upper_bound)
            out = out - zero_values

        out = torch.clamp(out, min=self.lower_bound, max=self.upper_bound)

        return out

    def get_l2_lipschitz_bound(self):
        input_transform_lipschitz = torch.max(torch.abs(self.input_transform)).item()
        return self.gamma * input_transform_lipschitz

# class NeuralNetwork(nn.Module):
#     def __init__(self, in_features, out_features, activations, widths, zero_at_zero=False, input_bias=None, input_transform=None, dtype=torch.float32):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.activations = activations  # List of activation function names
#         self.widths = widths  # List of widths for each layer
#         self.zero_at_zero = zero_at_zero
#         if input_bias is None:
#             self.input_bias = torch.zeros(in_features, dtype=dtype, requires_grad=False)
#         else:
#             self.input_bias = torch.tensor(input_bias, dtype=dtype, requires_grad=False)
#         if input_transform is None:
#             self.input_transform = torch.ones(in_features, dtype=dtype, requires_grad=False)
#         else:
#             self.input_transform = torch.tensor(input_transform, dtype=dtype, requires_grad=False)

#         if len(self.activations) != len(self.widths) - 2:
#             raise ValueError("Number of activations must be two less than number of widths. The last layer has no activation.")
#         if self.widths[-1] != self.out_features:
#             raise ValueError("Last width must match number of output channels.")
#         if self.widths[0] != self.in_features:
#             raise ValueError("First width must match number of input channels.")

#         layers = []
#         for i in range(len(self.activations)):
#             layers.append(nn.Linear(self.widths[i], self.widths[i + 1], bias=True))
#             layers.append(get_activation(self.activations[i]))
#         layers.append(nn.Linear(self.widths[-2], self.out_features))  # Final layer without activation
#         self.model = nn.Sequential(*layers)
            
#     def forward(self, x_in):
#         x = (x_in-self.input_bias) * self.input_transform
#         out = self.model(x)

#         if self.zero_at_zero:
#             zeros = torch.zeros_like(x_in)
#             zeros = (zeros-self.input_bias) * self.input_transform
#             zero_values = self.model(zeros)
#             out = out - zero_values
            
#         return out