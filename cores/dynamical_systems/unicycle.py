import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union
import math

def max_abs_sin(a, b):
    """
    Returns the maximum value of |sin(x)| for x in the interval [a, b].
    """
    # Check if there's an integer k such that x = π/2 + kπ lies in [a, b]
    # Solve for k: a ≤ π/2 + kπ ≤ b  ⟹  (a - π/2)/π ≤ k ≤ (b - π/2)/π
    lower_bound = (a - math.pi/2) / math.pi
    upper_bound = (b - math.pi/2) / math.pi

    # The smallest integer >= lower_bound
    k_candidate = math.ceil(lower_bound)

    # Check if this candidate point lies within [a, b]
    if math.pi/2 + k_candidate * math.pi <= b:
        return 1.0  # |sin(x)| reaches 1 at this point

    # Otherwise, the maximum is at one of the endpoints
    return max(abs(math.sin(a)), abs(math.sin(b)))

def max_abs_cos(a, b):
    """
    Returns the maximum value of |cos(x)| for x in the interval [a, b].
    """
    # Check if there's an integer k such that x = kπ lies in [a, b]
    # Solve for k: a ≤ kπ ≤ b  ⟹  a/π ≤ k ≤ b/π
    lower_bound = a / math.pi
    upper_bound = b / math.pi

    # The smallest integer >= lower_bound
    k_candidate = math.ceil(lower_bound)

    # Check if this candidate point lies within [a, b]
    if k_candidate * math.pi <= b:
        return 1.0  # |cos(x)| reaches 1 at this point

    # Otherwise, the maximum is at one of the endpoints
    return max(abs(math.cos(a)), abs(math.cos(b)))

class Unicycle(nn.Module):

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:

        super(Unicycle, self).__init__()

        self.state_dim = 3
        self.control_dim = 2
        self.dtype = dtype

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:

        dx = torch.zeros_like(x)
        theta = x[:, 2]
        v = u[:, 0]
        w = u[:, 1]

        dx[:, 0] = v * torch.cos(theta)
        dx[:, 1] = v * torch.sin(theta)
        dx[:, 2] = w
        return dx
    
    def get_drift(self, x: torch.Tensor) -> torch.Tensor:
        
        return torch.zeros_like(x)
    
    def get_actuation(self, x: torch.Tensor) -> torch.Tensor:
        
        actuation = torch.zeros(x.shape[0], x.shape[1], self.control_dim, dtype=self.dtype, device=x.device)
        theta = x[:, 2]
        actuation[:, 0, 0] = torch.cos(theta)
        actuation[:, 1, 0] = torch.sin(theta)
        actuation[:, 2, 1] = 1.0

        return actuation
    
    def f_dx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:

        f_dx = torch.zeros(x.shape[0], x.shape[1], x.shape[1], dtype=self.dtype, device=x.device)
        theta = x[:, 2]
        v = u[:, 0]
        f_dx[:, 0, 2] = -v * torch.sin(theta)
        f_dx[:, 1, 2] = v * torch.cos(theta)
        
        return f_dx
    
    def get_f_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        v_bound = max(abs(u_lb[0]), abs(u_ub[0]))
        w_bound = max(abs(u_lb[1]), abs(u_ub[1]))
        sin_theta_bound = max_abs_sin(x_lb[2], x_ub[2])
        cos_theta_bound = max_abs_cos(x_lb[2], x_ub[2])

        f_1_bound = v_bound * cos_theta_bound
        f_2_bound = v_bound * sin_theta_bound
        f_3_bound = w_bound
        
        f_bound = np.sqrt(f_1_bound**2 + f_2_bound**2 + f_3_bound**2)

        return f_bound
    
    def get_f_l1_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        v_bound = max(abs(u_lb[0]), abs(u_ub[0]))
        w_bound = max(abs(u_lb[1]), abs(u_ub[1]))
        sin_theta_bound = max_abs_sin(x_lb[2], x_ub[2])
        cos_theta_bound = max_abs_cos(x_lb[2], x_ub[2])

        f_1_bound = v_bound * cos_theta_bound
        f_2_bound = v_bound * sin_theta_bound
        f_3_bound = w_bound
        
        f_bound = f_1_bound + f_2_bound + f_3_bound

        return f_bound
    
    def get_f_dx_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        v_bound = max(abs(u_lb[0]), abs(u_ub[0]))
        sin_theta_bound = max_abs_sin(x_lb[2], x_ub[2])
        cos_theta_bound = max_abs_cos(x_lb[2], x_ub[2])

        f_dx = np.zeros((self.state_dim, self.state_dim))
        f_dx[0, 2] = v_bound * sin_theta_bound
        f_dx[1, 2] = v_bound * cos_theta_bound

        return np.linalg.norm(f_dx, ord=2)
    
    def get_f_dxdx_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> torch.Tensor:

        v_bound = max(abs(u_lb[0]), abs(u_ub[0]))
        sin_theta_bound = max_abs_sin(x_lb[2], x_ub[2])
        cos_theta_bound = max_abs_cos(x_lb[2], x_ub[2])

        f1_dxdx_bound = v_bound * cos_theta_bound
        f2_dxdx_bound = v_bound * sin_theta_bound
        f3_dxdx_bound = 0.0

        return torch.tensor([f1_dxdx_bound, f2_dxdx_bound, f3_dxdx_bound], dtype=self.dtype)