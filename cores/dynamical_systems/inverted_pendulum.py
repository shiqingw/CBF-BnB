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


class InvertedPendulum(nn.Module):
    """
    A PyTorch module implementing the dynamics of an inverted pendulum system.
    
    This class models the continuous-time dynamics of an inverted pendulum, including:
    - Forward dynamics computation
    - System linearization around the upright equilibrium
    - Bounds on dynamics and their derivatives for control purposes
    
    The state vector is x = [θ, θ̇], where:
        θ: angle from the upright position (radians)
        θ̇: angular velocity (radians/second)
    
    The control input u is the torque applied to the pendulum.
    
    Args:
        mass (float): Mass of the pendulum (kg)
        length (float): Length of the pendulum (m)
        viscous_friction (float): Coefficient of viscous friction
        dtype (torch.dtype, optional): Data type for torch tensors. Defaults to torch.float32.
    
    Attributes:
        state_dim (int): Dimension of the state vector (2)
        control_dim (int): Dimension of the control input (1)
        mass (torch.Tensor): Mass of the pendulum
        length (torch.Tensor): Length of the pendulum
        viscous_friction (torch.Tensor): Viscous friction coefficient
        gravity (torch.Tensor): Gravitational acceleration (9.81 m/s²)
    """
    
    def __init__(self, mass: float, length: float, viscous_friction: float, 
                 dtype: torch.dtype = torch.float32) -> None:
        super(InvertedPendulum, self).__init__()

        self.state_dim = 2
        self.control_dim = 1
        self.dtype = dtype

        self.register_buffer('mass', torch.tensor(mass, dtype=self.dtype))
        self.register_buffer('length', torch.tensor(length, dtype=self.dtype))
        self.register_buffer('viscous_friction', torch.tensor(viscous_friction, dtype=self.dtype))
        self.register_buffer('gravity', torch.tensor(9.81, dtype=self.dtype))
        
    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Computes the continuous-time dynamics of the inverted pendulum.

        Args:
            x (torch.Tensor): State vector [θ, θ̇], shape = (batch_size, 2)
            u (torch.Tensor): Control input (torque), shape = (batch_size, 1)

        Returns:
            torch.Tensor: State derivative [θ̇, θ̈], shape = (batch_size, 2)
        """
        theta, dtheta = x[:, 0:1], x[:, 1:2]

        ddtheta = self.gravity/self.length * torch.sin(theta)
        ddtheta += u / (self.mass * self.length**2)
        ddtheta -= self.viscous_friction / (self.mass * self.length**2) * dtheta

        dx = torch.cat([dtheta, ddtheta], dim=1)
        return dx
    
    def get_drift(self, x):

        theta, dtheta = x[:, 0:1], x[:, 1:2]

        ddtheta = self.gravity/self.length * torch.sin(theta)
        ddtheta -= self.viscous_friction / (self.mass * self.length**2) * dtheta

        dx = torch.cat([dtheta, ddtheta], dim=1)
        
        return dx
    
    def get_actuation(self, x):

        actuation = torch.zeros(x.shape[0], self.state_dim, self.control_dim, dtype=self.dtype, device=x.device)
        actuation[:, 1, 0] = 1 / (self.mass * self.length**2)

        return actuation
        
    def linearize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Linearizes the dynamics around the upright position: ẋ = Ax + Bu.

        The linearization is performed around the unstable equilibrium point
        x = [0, 0] (upright position with zero velocity) and u = 0.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - A: State matrix, shape = (2, 2)
                - B: Input matrix, shape = (2, 1)
        """
        gravity = self.gravity.item()
        mass = self.mass.item()
        length = self.length.item()
        friction = self.viscous_friction.item()

        A = np.array([[0, 1],
                      [gravity / length, -friction / (mass * length**2)]])

        B = np.array([[0],
                      [1 / (mass * length**2)]])

        return A, B
    
    def get_f_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:
        
        gravity = self.gravity.item()
        mass = self.mass.item()
        length = self.length.item()
        friction = self.viscous_friction.item()

        sin_theta_bound = max_abs_sin(x_lb[0], x_ub[0])
        dtheta_bound = max(abs(x_lb[1]), abs(x_ub[1]))
        u_bound = max(abs(u_lb[0]), abs(u_ub[0]))

        f_1_bound = dtheta_bound
        f_2_bound = sin_theta_bound*gravity/length + u_bound/(mass*length**2) + friction/(mass*length**2)*dtheta_bound
        
        f_bound = np.sqrt(f_1_bound**2 + f_2_bound**2)

        return f_bound
    
    def get_f_du_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:
        """Calculates the L2 norm bound of ∂f/∂u (partial derivative of dynamics w.r.t. control).

        Returns:
            float: Upper bound on the L2 norm of ∂f/∂u
        """
        mass = self.mass.item()
        length = self.length.item()

        df_du_bound = 1 / (mass * length**2)
        return df_du_bound
    
    def get_f_dx_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:
        """Calculates the L2 norm bound of ∂f/∂x (partial derivative of dynamics w.r.t. state).

        Returns:
            float: Upper bound on the L2 norm of ∂f/∂x
        """
        gravity = self.gravity.item()
        mass = self.mass.item()
        length = self.length.item()
        friction = self.viscous_friction.item()

        cos_theta_bound = max_abs_cos(x_lb[0], x_ub[0])

        df_dx = np.array([[0, 1],
                          [gravity / length * cos_theta_bound, friction / (mass * length**2)]], dtype=np.float32)

        return np.linalg.norm(df_dx, ord=2)
    
    def f_dx(self, x, u):

        theta = x[:, 0]

        gravity = self.gravity.item()
        mass = self.mass.item()
        length = self.length.item()
        friction = self.viscous_friction.item()

        N = x.shape[0]
        f_dx = torch.zeros(N, self.state_dim, self.state_dim, dtype=self.dtype, device=x.device) # (N, 2, 2)
        f_dx[:, 0, 1] = 1.0
        f_dx[:, 1, 0] = gravity / length * torch.cos(theta)
        f_dx[:, 1, 1] = -friction / (mass * length**2)

        return f_dx
    
    def f_du(self, x, u):

        mass = self.mass.item()
        length = self.length.item()

        N = x.shape[0]
        f_du = torch.zeros(N, self.state_dim, self.control_dim, dtype=self.dtype, device=x.device)
        f_du[:, 1, 0] = 1 / (mass * length**2)

        return f_du
    
    def get_f_dxdx_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):

        length = self.length.item()
        gravity = self.gravity.item()

        f_dxdx = torch.zeros(self.state_dim, self.state_dim, self.state_dim, dtype=self.dtype)
        sin_theta_bound = max_abs_sin(x_lb[0], x_ub[0])

        f_dxdx[1, 1, 0] = gravity / length * sin_theta_bound

        f_dxdx_elementwise_l2_bound = torch.linalg.norm(f_dxdx, ord=2, dim=(1,2)) # (2,)
        return f_dxdx_elementwise_l2_bound

    def get_f_dxdu_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):

        return torch.zeros(self.state_dim, dtype=self.dtype)
    
    def get_f_dudu_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):
        
        return torch.zeros(self.state_dim, dtype=self.dtype)
    