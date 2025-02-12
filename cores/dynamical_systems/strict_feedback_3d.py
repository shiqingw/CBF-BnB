import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union

def max_alpha_beta_expression(a1, b1, a2, b2, alpha, beta):
    """
    Computes the maximum of |alpha * x1 + beta * x2|
    where x1 is in [a1, b1] and x2 is in [a2, b2].

    Parameters:
    a1 (float): Lower bound for x1
    b1 (float): Upper bound for x1
    a2 (float): Lower bound for x2
    b2 (float): Upper bound for x2
    alpha (float): Coefficient for x1
    beta (float): Coefficient for x2

    Returns:
    float: The maximum absolute value of (alpha * x1 + beta * x2)
    """
    # Define the four corner points
    points = [
        (a1, a2),
        (a1, b2),
        (b1, a2),
        (b1, b2)
    ]
    
    # Compute |alpha * x1 + beta * x2| for each corner
    max_val = max(abs(alpha * x1 + beta * x2) for x1, x2 in points)
    
    return max_val

class StrictFeedback3D(nn.Module):
    """
    A neural network module representing a strict feedback 3D system.

    This module defines a dynamical system with state dimension 3 and control dimension 1.
    The system dynamics are defined by parameters `a1`, `a2`, `b1`, `b2`, `c1`, and `c2`.

    Attributes:
        state_dim (int): Dimension of the state vector (3).
        control_dim (int): Dimension of the control vector (1).
        dtype (torch.dtype): Data type for tensors.
        a1 (torch.Tensor): Parameter `a1`.
        a2 (torch.Tensor): Parameter `a2`.
        b1 (torch.Tensor): Parameter `b1`.
        b2 (torch.Tensor): Parameter `b2`.
        c1 (torch.Tensor): Parameter `c1`.
        c2 (torch.Tensor): Parameter `c2`.
    """

    def __init__(self, a1: float, a2: float, b1: float, b2: float, c1: float, c2: float,
                 dtype: torch.dtype = torch.float32) -> None:
        """
        Initialize the StricFeedback3D module with system parameters.

        Args:
            a1 (float): Parameter `a1`.
            a2 (float): Parameter `a2`.
            b1 (float): Parameter `b1`.
            b2 (float): Parameter `b2`.
            c1 (float): Parameter `c1`.
            c2 (float): Parameter `c2`.
            dtype (torch.dtype, optional): Data type for tensors. Defaults to `torch.float32`.
        """

        super(StrictFeedback3D, self).__init__()

        self.state_dim = 3
        self.control_dim = 1
        self.dtype = dtype

        self.register_buffer('a1', torch.tensor(a1, dtype=self.dtype))
        self.register_buffer('a2', torch.tensor(a2, dtype=self.dtype))
        self.register_buffer('b1', torch.tensor(b1, dtype=self.dtype))
        self.register_buffer('b2', torch.tensor(b2, dtype=self.dtype))
        self.register_buffer('c1', torch.tensor(c1, dtype=self.dtype))
        self.register_buffer('c2', torch.tensor(c2, dtype=self.dtype))

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute the state derivatives given the current state and control input.

        Args:
            x (torch.Tensor): Current state tensor of shape `(batch_size, 3)`.
            u (torch.Tensor): Control input tensor of shape `(batch_size, 1)`.

        Returns:
            torch.Tensor: State derivatives tensor of shape `(batch_size, 3)`.
        """

        x1, x2, x3 = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        a1 = self.a1
        a2 = self.a2
        b1 = self.b1
        b2 = self.b2
        c1 = self.c1
        c2 = self.c2

        dx1 = a1 * x1 + a2 * x2
        dx2 = b1 * x2 + b2 * x3
        dx3 = c1 * x3**2 + c2 * u

        dx = torch.cat((dx1, dx2, dx3), dim=1)

        return dx
    
    def linearize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the system around the origin.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the linearized system matrices `(A, B)`.
                - `A` (np.ndarray): The system matrix of shape `(3, 3)`.
                - `B` (np.ndarray): The input matrix of shape `(3, 1)`.
        """

        a1 = self.a1.item()
        a2 = self.a2.item()
        b1 = self.b1.item()
        b2 = self.b2.item()
        c2 = self.c2.item()

        A = np.array([[a1, a2, 0],
                      [0, b1, b2],
                      [0, 0, 0]], dtype=np.float32)
        
        B = np.array([[0],
                      [0],
                      [c2]], dtype=np.float32)
        
        return A, B
    
    def get_f_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        a1 = self.a1.item()
        a2 = self.a2.item()
        b1 = self.b1.item()
        b2 = self.b2.item()
        c1 = self.c1.item()
        c2 = self.c2.item()

        f1_bound = max_alpha_beta_expression(x_lb[0], x_ub[0], x_lb[1], x_ub[1], a1, a2)
        f2_bound = max_alpha_beta_expression(x_lb[1], x_ub[1], x_lb[2], x_ub[2], b1, b2)

        x3_squared_bound_lb = max(abs(x_lb[2])**2, abs(x_ub[2])**2)
        x3_squared_bound_ub = max(abs(x_lb[2])**2, abs(x_ub[2])**2)
        f3_bound = max_alpha_beta_expression(x3_squared_bound_lb, x3_squared_bound_ub, u_lb[0], u_ub[0], c1, c2)

        f_bound = np.sqrt(f1_bound**2 + f2_bound**2 + f3_bound**2)

        return f_bound
    
    def get_f_du_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:
        """
        Compute the L2-norm bound of the partial derivative of `f` with respect to `u`.

        Returns:
            float: The L2-norm bound of ∂f/∂u.
        """

        c2 = self.c2.item()

        return abs(c2)
    
    def get_f_dx_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:
        """
        Compute the L2-norm bound of the Jacobian of `f` with respect to `x` over a given bound of `x3`.

        Args:
            x3_bound (float): Upper bound for the absolute value of state `x3`.

        Returns:
            float: The L2-norm bound of ∂f/∂x.
        """

        x3_bound  = max(abs(x_lb[2]), abs(x_ub[2]))
        
        a1 = self.a1.item()
        a2 = self.a2.item()
        b1 = self.b1.item()
        b2 = self.b2.item()
        c1 = self.c1.item()

        df_dx = np.zeros((3, 3), dtype=np.float32)
        df_dx[0, 0] = abs(a1)
        df_dx[0, 1] = abs(a2)
        df_dx[1, 1] = abs(b1)
        df_dx[1, 2] = abs(b2)
        df_dx[2, 2] = 2*abs(c1)*x3_bound

        return np.linalg.norm(df_dx, ord=2)
    
    def f_dx(self, x, u):

        x3 = x[:, 2]

        a1 = self.a1.item()
        a2 = self.a2.item()
        b1 = self.b1.item()
        b2 = self.b2.item()
        c1 = self.c1.item()

        N = x.shape[0]
        f_dx = torch.zeros(N, self.state_dim, self.state_dim, dtype=self.dtype, device=x.device) # (N, 3, 3)
        f_dx[:, 0, 0] = a1
        f_dx[:, 0, 1] = a2
        f_dx[:, 1, 1] = b1
        f_dx[:, 1, 2] = b2
        f_dx[:, 2, 2] = 2*c1*x3

        return f_dx
    
    def f_du(self, x, u):

        c2 = self.c2.item()

        N = x.shape[0]
        f_du = torch.zeros(N, self.state_dim, self.control_dim, dtype=self.dtype, device=x.device)
        f_du[:, 2, 0] = c2

        return f_du
    
    def get_f_dxdx_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):

        c1 = self.c1.item()

        f_dxdx = torch.zeros(self.state_dim, self.state_dim, self.state_dim, dtype=self.dtype)
        f_dxdx[2, 2, 2] = 2*c1

        f_dxdx_elementwise_l2_bound = torch.linalg.norm(f_dxdx, ord=2, dim=(1,2)) # (2,)
        return f_dxdx_elementwise_l2_bound

    def get_f_dxdu_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):

        return torch.zeros(self.state_dim, dtype=self.dtype)
    
    def get_f_dudu_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):
        
        return torch.zeros(self.state_dim, dtype=self.dtype)