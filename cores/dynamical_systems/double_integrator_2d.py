import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union
import math

class DoubleIntegrator2D(nn.Module):

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:

        super(DoubleIntegrator2D, self).__init__()

        self.state_dim = 4
        self.control_dim = 2
        self.dtype = dtype

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:

        dx = torch.zeros_like(x)
        vx = x[:,2]
        vy = x[:,3]
        ax = u[:,0]
        ay = u[:,1]

        dx[:, 0] = vx
        dx[:, 1] = vy
        dx[:, 2] = ax
        dx[:, 3] = ay
        return dx
    
    def get_drift(self, x: torch.Tensor) -> torch.Tensor:

        vx = x[:,2]
        vy = x[:,3]

        drift = torch.zeros_like(x)
        drift[:, 0] = vx
        drift[:, 1] = vy
        
        return drift
    
    def get_actuation(self, x: torch.Tensor) -> torch.Tensor:
        
        actuation = torch.zeros(x.shape[0], x.shape[1], self.control_dim, dtype=self.dtype, device=x.device)
        actuation[:, 2, 0] = 1.0
        actuation[:, 3, 1] = 1.0

        return actuation
    
    def f_dx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:

        f_dx = torch.zeros(x.shape[0], x.shape[1], x.shape[1], dtype=self.dtype, device=x.device)
        f_dx[:, 0, 2] = 1.0
        f_dx[:, 1, 3] = 1.0

        return f_dx
    
    def get_f_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        vx_bound = max(abs(x_lb[2]), abs(x_ub[2]))
        vy_bound = max(abs(x_lb[3]), abs(x_ub[3]))
        ax_bound = max(abs(u_lb[0]), abs(u_ub[0]))
        ay_bound = max(abs(u_lb[1]), abs(u_ub[1]))

        f_1_bound = vx_bound
        f_2_bound = vy_bound
        f_3_bound = ax_bound
        f_4_bound = ay_bound
        
        f_bound = np.sqrt(f_1_bound**2 + f_2_bound**2 + f_3_bound**2 + f_4_bound**2)

        return f_bound
    
    def get_f_l1_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        vx_bound = max(abs(x_lb[2]), abs(x_ub[2]))
        vy_bound = max(abs(x_lb[3]), abs(x_ub[3]))
        ax_bound = max(abs(u_lb[0]), abs(u_ub[0]))
        ay_bound = max(abs(u_lb[1]), abs(u_ub[1]))

        f_1_bound = vx_bound
        f_2_bound = vy_bound
        f_3_bound = ax_bound
        f_4_bound = ay_bound
        
        f_bound = f_1_bound + f_2_bound + f_3_bound + f_4_bound

        return f_bound
    
    def get_f_dx_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        f_dx = np.zeros((self.state_dim, self.state_dim))
        f_dx[0, 2] = 1.0
        f_dx[1, 3] = 1.0

        return np.linalg.norm(f_dx, ord=2)
    
    def get_f_dxdx_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> torch.Tensor:

        return torch.zeros(self.state_dim, dtype=self.dtype)