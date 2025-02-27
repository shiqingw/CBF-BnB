import torch
import torch.nn as nn
import numpy as np

class SingleIntegrator3D(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32):
        super(SingleIntegrator3D, self).__init__()

        self.state_dim = 3
        self.control_dim = 3
        self.dtype = dtype

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:

        return u
    
    def get_drift(self, x: torch.Tensor) -> torch.Tensor:
        
        return torch.zeros_like(x)
    
    def get_actuation(self, x: torch.Tensor) -> torch.Tensor:
        
        actuation = torch.zeros(x.shape[0], x.shape[1], self.control_dim, dtype=self.dtype, device=x.device)
        actuation[:, 0, 0] = 1.0
        actuation[:, 1, 1] = 1.0
        actuation[:, 2, 2] = 1.0

        return actuation
    
    def f_dx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:

        f_dx = torch.zeros(x.shape[0], x.shape[1], x.shape[1], dtype=self.dtype, device=x.device)

        return f_dx
    
    def get_f_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        f_1_bound = max(abs(u_lb[0]), abs(u_ub[0]))
        f_2_bound = max(abs(u_lb[1]), abs(u_ub[1]))
        f_3_bound = max(abs(u_lb[2]), abs(u_ub[2]))
        
        f_bound = np.sqrt(f_1_bound**2 + f_2_bound**2 + f_3_bound**2)

        return f_bound
    
    def get_f_l1_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        f_1_bound = max(abs(u_lb[0]), abs(u_ub[0]))
        f_2_bound = max(abs(u_lb[1]), abs(u_ub[1]))
        f_3_bound = max(abs(u_lb[2]), abs(u_ub[2]))
        
        f_bound = f_1_bound + f_2_bound + f_3_bound

        return f_bound
    
    def get_f_dx_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        f_dx = np.zeros((self.state_dim, self.state_dim))

        return np.linalg.norm(f_dx, ord=2)
    
    def get_f_dxdx_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> torch.Tensor:

        return torch.zeros(self.state_dim, dtype=self.dtype)