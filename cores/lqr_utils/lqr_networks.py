import torch 
import torch.nn as nn 

class QuadraticFunction(nn.Module):
    def __init__(self, in_features, P_matrix, dtype=torch.float32):
        super(QuadraticFunction, self).__init__()
        self.P = nn.Parameter(P_matrix.to(dtype))
        self.in_features = in_features
        self.dtype = dtype

    def forward(self, x):
        return (x @ self.P * x).sum(dim=1, keepdim=True)
    
    def forward_with_jacobian(self, x):
        return (x @ self.P * x).sum(dim=1, keepdim=True), (2 * x @ self.P).unsqueeze(1)
    
class LinearFunction(nn.Module):
    def __init__(self, in_features, out_features, K_matrix, lower_bound=None, upper_bound=None, dtype=torch.float32):
        super(LinearFunction, self).__init__()
        self.K = nn.Parameter(K_matrix.to(dtype))
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

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

    def forward(self, x):
        out = x @ self.K
        
        if self.lower_bound is not None:
            out = torch.max(out, self.lower_bound)
        if self.upper_bound is not None:
            out = torch.min(out, self.upper_bound)
        
        return out