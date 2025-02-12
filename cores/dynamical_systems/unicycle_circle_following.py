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

class UnicycleCircleFollowing(nn.Module):
    """
    A PyTorch module for modeling a unicycle robot following a circular path.

    This module defines the dynamics of a unicycle robot as it follows a circular path.
    It provides methods to compute the state derivatives, linearize the system,
    and calculate bounds on the dynamics and their derivatives.
    """

    def __init__(self, path_radius: float, linear_velocity: float, dtype: torch.dtype = torch.float32) -> None:
        """
        Initialize the UnicycleCircleFollowing module.

        Args:
            path_radius (float): Radius of the circular path to follow.
            linear_velocity (float): Constant linear velocity of the unicycle robot.
            dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
        """

        super(UnicycleCircleFollowing, self).__init__()

        self.state_dim = 2
        self.control_dim = 1
        self.dtype = dtype

        self.register_buffer('path_radius', torch.tensor(path_radius, dtype=self.dtype))
        self.register_buffer('linear_velocity', torch.tensor(linear_velocity, dtype=self.dtype))
    
    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute the time derivative of the state.

        Args:
            x (torch.Tensor): State tensor of shape (batch_size, 2), containing [dist_e, theta_e].
            u (torch.Tensor): Control input tensor of shape (batch_size, 1).

        Returns:
            torch.Tensor: State derivative tensor of shape (batch_size, 2).
        """

        v = self.linear_velocity
        R = self.path_radius

        dist_e, theta_e = x[:, 0:1], x[:, 1:2]

        sin_theta_e = torch.sin(theta_e)
        cos_theta_e = torch.cos(theta_e)

        d_dist_e = v * sin_theta_e
        d_theta_e = u - v * cos_theta_e / (R - dist_e) + v / R

        dx = torch.cat([d_dist_e, d_theta_e], dim=1)
        return dx
    
    def linearize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the dynamics around the nominal trajectory.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the linearized system matrices (A, B).
                - A (np.ndarray): State matrix of shape (2, 2).
                - B (np.ndarray): Input matrix of shape (2, 1).
        """

        v = self.linear_velocity.item()
        R = self.path_radius.item()
    
        A = np.array([[0, v],
                      [v/R**2, 0]], dtype=np.float32)
        
        B = np.array([[0],
                      [1]], dtype=np.float32)
        
        return A, B
    
    def get_f_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        v = self.linear_velocity.item()
        R = self.path_radius.item()

        assert R - x_ub[0] > 0, "R - x_ub[0] must be greater than zero"

        sin_theta_e_bound = max_abs_sin(x_lb[1], x_ub[1])
        cos_theta_e_bound = max_abs_cos(x_lb[1], x_ub[1])
        R_minus_d_e_min = R - x_ub[0]
        u_bound = max(abs(u_lb[0]), abs(u_ub[0]))

        f_1_bound = v * sin_theta_e_bound

        f_2_bound = u_bound + v * cos_theta_e_bound/ R_minus_d_e_min + v / R

        f_bound = np.sqrt(f_1_bound**2 + f_2_bound**2)
        
        return f_bound
    
    def get_f_du_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:
        """
        Compute an upper bound on the L2 norm of the partial derivative of f with respect to u.

        Returns:
            float: Upper bound on the L2 norm of ∂f/∂u.
        """

        return 1.0
    
    def get_f_dx_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:
        """
        Compute an upper bound on the L2 norm of the partial derivative of f with respect to x.

        Returns:
            float: Upper bound on the L2 norm of ∂f/∂x.

        Raises:
            AssertionError: If any of the bounds are invalid.
        """

        v = self.linear_velocity.item()
        R = self.path_radius.item()

        assert R - x_ub[0] > 0, "R - x_ub[0] must be greater than zero"

        sin_theta_e_bound = max_abs_sin(x_lb[1], x_ub[1])
        cos_theta_e_bound = max_abs_cos(x_lb[1], x_ub[1])
        R_minus_d_e_min = R - x_ub[0]

        f_dx_bound = np.zeros((self.state_dim, self.state_dim), dtype=np.float32)
        f_dx_bound[0, 1] = v * cos_theta_e_bound
        f_dx_bound[1, 0] = v * cos_theta_e_bound / R_minus_d_e_min**2
        f_dx_bound[1, 1] = v * sin_theta_e_bound / R_minus_d_e_min

        return np.linalg.norm(f_dx_bound, ord=2)
    
    def f_dx(self, x, u):

        d_e = x[:, 0]
        theta_e = x[:, 1]

        v = self.linear_velocity.item()
        R = self.path_radius.item()

        N = x.shape[0]
        f_dx = torch.zeros(N, self.state_dim, self.state_dim, dtype=self.dtype, device=x.device) # (N, 2, 2)
        f_dx[:, 0, 1] = v * torch.cos(theta_e)
        f_dx[:, 1, 0] = - v * torch.cos(theta_e) / (R - d_e)**2
        f_dx[:, 1, 1] = v * torch.sin(theta_e) / (R - d_e)

        return f_dx
    
    def f_du(self, x, u):

        N = x.shape[0]
        f_du = torch.zeros(N, self.state_dim, self.control_dim, dtype=self.dtype, device=x.device)
        f_du[:, 1, 0] = 1.0

        return f_du
    
    def get_f_dxdx_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):

        v = self.linear_velocity.item()
        R = self.path_radius.item()

        assert R - x_ub[0] > 0, "R - x_ub[0] must be greater than zero"

        f_dxdx = torch.zeros(self.state_dim, self.state_dim, self.state_dim, dtype=self.dtype)
        sin_theta_bound = max_abs_sin(x_lb[1], x_ub[1])
        cos_theta_bound = max_abs_cos(x_lb[1], x_ub[1])
        R_minus_d_e_min = float(R - x_ub[0])

        f_dxdx[0, 1, 1] = v * sin_theta_bound
        f_dxdx[1, 0, 0] = 2 * v * cos_theta_bound / R_minus_d_e_min**3
        f_dxdx[1, 0, 1] = v * sin_theta_bound / R_minus_d_e_min**2
        f_dxdx[1, 1, 0] = v * sin_theta_bound / R_minus_d_e_min**2
        f_dxdx[1, 1, 1] = v * cos_theta_bound / R_minus_d_e_min

        f_dxdx_elementwise_l2_bound = torch.linalg.norm(f_dxdx, ord=2, dim=(1,2)) # (2,)
        return f_dxdx_elementwise_l2_bound

    def get_f_dxdu_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):

        return torch.zeros(self.state_dim, dtype=self.dtype)
    
    def get_f_dudu_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):
        
        return torch.zeros(self.state_dim, dtype=self.dtype)