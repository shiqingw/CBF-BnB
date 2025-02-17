import torch
import torch.nn as nn
import torch.nn.functional as F

## from https://github.com/locuslab/orthogonal-convolutions
def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape 
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:] # W = [U, V]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V # A = U - U^T + V^T V
    iIpA = torch.inverse(I + A)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)

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
    
def get_activation_second_der(activation_name):
    if activation_name == 'relu':
        def relu_second_derivative(x):
            return torch.zeros_like(x)
        return relu_second_derivative
    
    elif activation_name == 'sigmoid':
        def sigmoid_second_derivative(x):
            sig = torch.sigmoid(x)
            return sig * (1 - sig) * (1 - 2 * sig)
        return sigmoid_second_derivative

    elif activation_name == 'tanh':
        def tanh_second_derivative(x):
            return -2 * torch.tanh(x) * (1 - torch.tanh(x) ** 2)
        return tanh_second_derivative
    
    elif activation_name == 'softplus':
        def softplus_second_derivative(x):
            return torch.sigmoid(x) * (1 - torch.sigmoid(x))
        return softplus_second_derivative

    elif activation_name == 'identity':
        def identity_second_derivative(x):
            return torch.zeros_like(x)
        return identity_second_derivative
    
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

## from https://github.com/acfr/LBDN
class SandwichFc(nn.Linear): 
    def __init__(self, in_features, out_features, bias=True, activation='relu', scale=1.0, dtype=torch.float32, 
                random_psi=False, trainable_psi=True, initialize_all_weights_to_zero=False):
        super().__init__(in_features+out_features, out_features, bias, dtype=dtype)
        if initialize_all_weights_to_zero:
            self.weight.data = torch.zeros_like(self.weight)
            self.bias.data = torch.zeros_like(self.bias)
        self.dtype = dtype
        self.alpha = nn.Parameter(torch.ones(1, dtype=self.dtype, requires_grad=True))
        self.alpha.data = self.weight.norm() 
        self.scale = scale 
        if random_psi:
            self.psi = nn.Parameter(torch.rand(out_features, dtype=self.dtype)-0.5, requires_grad=trainable_psi)
        else:
            self.psi = nn.Parameter(torch.zeros(out_features, dtype=self.dtype), requires_grad=trainable_psi)
        self.Q = None
        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.activation_der = get_activation_der(activation)
        self.activation_second_der = get_activation_second_der(activation)
    
    def _apply(self, fn):
        super()._apply(fn)
        self.Q = None  # Reset Q whenever the module is moved to a new device
        return self
    
    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self.Q = None
        return self
    
    def forward(self, x): # Eq. (9)
        if x.dim() != 2:
            raise ValueError("Input tensor must be 2D")
        
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        
        # Detach parameters in eval mode
        if self.training:
            Q = self.Q
            psi = self.psi
            bias = self.bias if self.bias is not None else None
        else:
            Q = self.Q.detach()
            psi = self.psi.detach()
            bias = self.bias.detach() if self.bias is not None else None

        # Calculate the output
        x = F.linear(self.scale * x, Q[:, fout:]) # B * h 
        x = x * torch.exp(-psi) * (2 ** 0.5) # sqrt(2) \Psi^{-1} B * h
        if bias is not None:
            x += bias
        x = self.activation(x) * torch.exp(psi) # \Psi z
        x = 2 ** 0.5 * F.linear(x, Q[:, :fout].T) # sqrt(2) A^top \Psi z
        return x
    
    def forward_with_jacobian(self, x):
        if x.dim() != 2:
            raise ValueError("Input tensor must be 2D")
        
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        
        # Detach parameters in eval mode
        if self.training:
            Q = self.Q
            psi = self.psi
            bias = self.bias if self.bias is not None else None
        else:
            Q = self.Q.detach()
            psi = self.psi.detach()
            bias = self.bias.detach() if self.bias is not None else None

        W1 = self.scale * (2 ** 0.5) * torch.diag(torch.exp(-psi)) @ Q[:, fout:]
        W2 = (2 ** 0.5) * Q[:, :fout].T @ torch.diag(torch.exp(psi))

        # Calculate the output
        out = F.linear(x, W1) # sqrt(2) \Psi^{-1} B * h
        if bias is not None:
            out += bias

        # Calculate the input jacobian
        jac = W2.unsqueeze(0) * self.activation_der(out).unsqueeze(1)
        jac = torch.matmul(jac, W1)

        # Continue calculating the output
        out = F.linear(self.activation(out), W2) # sqrt(2) A^top \Psi \sigma(z)

        return out, jac
    
    def forward_with_jacobian_and_hessian(self, x):
        if x.dim() != 2:
            raise ValueError("Input tensor must be 2D")
        
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        
        # Detach parameters in eval mode
        if self.training:
            Q = self.Q
            psi = self.psi
            bias = self.bias if self.bias is not None else None
        else:
            Q = self.Q.detach()
            psi = self.psi.detach()
            bias = self.bias.detach() if self.bias is not None else None

        W1 = self.scale * (2 ** 0.5) * torch.diag(torch.exp(-psi)) @ Q[:, fout:]
        W2 = (2 ** 0.5) * Q[:, :fout].T @ torch.diag(torch.exp(psi))

        # Calculate the output
        out = F.linear(x, W1) # sqrt(2) \Psi^{-1} B * h
        if bias is not None:
            out += bias

        # Calculate the input jacobian
        jac = W2.unsqueeze(0) * self.activation_der(out).unsqueeze(1)
        jac = torch.matmul(jac, W1)

        # Calculate the second derivative
        sigma_second_der = self.activation_second_der(out)
        hess = torch.einsum('ij,bj,jk,jl -> bikl', 
                            W2,
                            sigma_second_der, 
                            W1, 
                            W1)

        # Continue calculating the output
        out = F.linear(self.activation(out), W2) # sqrt(2) A^top \Psi \sigma(z)

        return out, jac, hess
    
    def forward_with_jacobian_and_hessian_method2(self, x):
        if x.dim() != 2:
            raise ValueError("Input tensor must be 2D")
        
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        
        # Detach parameters in eval mode
        if self.training:
            Q = self.Q
            psi = self.psi
            bias = self.bias if self.bias is not None else None
        else:
            Q = self.Q.detach()
            psi = self.psi.detach()
            bias = self.bias.detach() if self.bias is not None else None

        W1 = self.scale * (2 ** 0.5) * torch.diag(torch.exp(-psi)) @ Q[:, fout:]
        W2 = (2 ** 0.5) * Q[:, :fout].T @ torch.diag(torch.exp(psi))

        # Calculate the output
        out = F.linear(x, W1) # sqrt(2) \Psi^{-1} B * h
        if bias is not None:
            out += bias

        # Calculate the input jacobian
        jac = W2.unsqueeze(0) * self.activation_der(out).unsqueeze(1)
        jac = torch.matmul(jac, W1)

        # Calculate the second derivative
        sigma_second_der = self.activation_second_der(out)

        # Continue calculating the output
        out = F.linear(self.activation(out), W2) # sqrt(2) A^top \Psi \sigma(z)

        return out, jac, W1, W2, sigma_second_der
    
class SandwichLin(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, AB=False, dtype=torch.float32,
                 initialize_all_weights_to_zero=False):
        super().__init__(in_features+out_features, out_features, bias, dtype=dtype)
        if initialize_all_weights_to_zero:
            self.weight.data = torch.zeros_like(self.weight)
            self.bias.data = torch.zeros_like(self.bias)
        self.dtype = dtype
        self.alpha = nn.Parameter(torch.ones(1, dtype=self.dtype, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale   
        self.AB = AB
        self.Q = None
    
    def _apply(self, fn):
        super()._apply(fn)
        self.Q = None  # Reset Q whenever the module is moved to a new device
        return self
    
    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self.Q = None
        return self

    def forward(self, x): # Eq. (9)
        if x.dim() != 2:
            raise ValueError("Input tensor must be 2D")
        
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        
        if self.training:
            Q = self.Q
            bias = self.bias if self.bias is not None else None
        else:
            Q = self.Q.detach()
            bias = self.bias.detach() if self.bias is not None else None

        x = F.linear(self.scale * x, Q[:, fout:]) # B @ x 
        if self.AB:
            x = 2 * F.linear(x, Q[:, :fout].T) # 2 A.T @ B @ x
        if bias is not None:
            x += bias
        return x
    
    def forward_with_jacobian(self, x):
        if x.dim() != 2:
            raise ValueError("Input tensor must be 2D")
        
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        
        if self.training:
            Q = self.Q
            bias = self.bias if self.bias is not None else None
        else:
            Q = self.Q.detach()
            bias = self.bias.detach() if self.bias is not None else None

        W1 = self.scale * Q[:, fout:]
        W2 = 2 * Q[:, :fout].T

        out = F.linear(x, W1)
        jac = torch.ones_like(out).unsqueeze(2) * W1.unsqueeze(0)

        if self.AB:
            out = F.linear(out, W2)
            jac = torch.matmul(W2, jac)
        
        if bias is not None:
            out += bias

        return out, jac

    def forward_with_jacobian_and_hessian(self, x):
        if x.dim() != 2:
            raise ValueError("Input tensor must be 2D")
        
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        
        if self.training:
            Q = self.Q
            bias = self.bias if self.bias is not None else None
        else:
            Q = self.Q.detach()
            bias = self.bias.detach() if self.bias is not None else None

        W1 = self.scale * Q[:, fout:]
        W2 = 2 * Q[:, :fout].T

        out = F.linear(x, W1)
        jac = torch.ones_like(out).unsqueeze(2) * W1.unsqueeze(0)

        # Calculate the second derivative (should be zero)
        sigma_second_der = torch.zeros_like(out)

        if self.AB:
            out = F.linear(out, W2)
            jac = torch.matmul(W2, jac)
        else:
            W2 = torch.eye(fout, dtype=self.dtype, device=self.weight.device)

        hess = torch.einsum('ij,bj,jk,jl -> bikl', 
                            W2,
                            sigma_second_der, 
                            W1, 
                            W1)
        
        if bias is not None:
            out += bias

        return out, jac, hess
    
    def forward_with_jacobian_and_hessian_method2(self, x):
        if x.dim() != 2:
            raise ValueError("Input tensor must be 2D")
        
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        
        if self.training:
            Q = self.Q
            bias = self.bias if self.bias is not None else None
        else:
            Q = self.Q.detach()
            bias = self.bias.detach() if self.bias is not None else None

        W1 = self.scale * Q[:, fout:]
        W2 = 2 * Q[:, :fout].T

        out = F.linear(x, W1)
        jac = torch.ones_like(out).unsqueeze(2) * W1.unsqueeze(0)

        # Calculate the second derivative (should be zero)
        sigma_second_der = torch.zeros_like(out)

        if self.AB:
            out = F.linear(out, W2)
            jac = torch.matmul(W2, jac)
        else:
            W2 = torch.eye(fout, dtype=self.dtype, device=self.weight.device)
        
        if bias is not None:
            out += bias

        return out, jac, W1, W2, sigma_second_der