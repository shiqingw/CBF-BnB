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

def max_abs_cos_minus_one(a, b):
    """
    Returns the maximum value of |cos(x) - 1| for x in the interval [a, b].
    """
    # Check if there's an integer k such that x = (2k+1)π lies in [a, b].
    # Solve for k: a ≤ (2k+1)π ≤ b  ⟹  (a/π - 1)/2 ≤ k ≤ (b/π - 1)/2
    lower_bound = (a / math.pi - 1) / 2
    upper_bound = (b / math.pi - 1) / 2

    # The smallest integer greater than or equal to lower_bound
    k_candidate = math.ceil(lower_bound)

    # Candidate x value of the form (2k+1)π
    x_candidate = (2 * k_candidate + 1) * math.pi

    # Check if this candidate lies within [a, b]
    if a <= x_candidate <= b:
        # cos(x_candidate) = -1, so |cos(x_candidate) - 1| = 2
        return 2.0

    # If no such point exists in [a, b], check endpoints.
    return max(abs(math.cos(a) - 1.0), abs(math.cos(b) - 1.0))

class Quadrotor2D(nn.Module):
    def __init__(self, mass: float, inertia: float, arm_length: float,
                 dtype: torch.dtype = torch.float32) -> None:
        """Initialize the Quadrotor2D system parameters.

        Args:
            mass (float): Mass of the quadrotor.
            inertia (float): Moment of inertia of the quadrotor.
            arm_length (float): Length of the quadrotor arm.
            dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
        """

        super(Quadrotor2D, self).__init__()

        self.state_dim = 6
        self.control_dim = 2
        self.dtype = dtype

        self.register_buffer('mass', torch.tensor(mass, dtype=self.dtype))
        self.register_buffer('inertia', torch.tensor(inertia, dtype=self.dtype))
        self.register_buffer('arm_length', torch.tensor(arm_length, dtype=self.dtype))
        self.register_buffer('gravity', torch.tensor(9.81, dtype=self.dtype))

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Computes the continuous-time dynamics of the 2D quadrotor system.

        Args:
            x (torch.Tensor): State vector [x, y, θ, ẋ, ẏ, θ̇], shape = (batch_size, 6)
            u (torch.Tensor): Control input [u1, u2], shape = (batch_size, 2)

        Returns:
            torch.Tensor: State derivative [ẋ, ẏ, θ̇, ẍ, ÿ, θ̈], shape = (batch_size, 6)
        """
        assert x.shape[1] == self.state_dim, "Invalid state dimension."
        assert u.shape[1] == self.control_dim, "Invalid control dimension."

        theta, dx, dy, dtheta = x[:, 2:3], x[:, 3:4], x[:, 4:5], x[:, 5:6]
        u1, u2 = u[:, 0:1], u[:, 1:2]

        mass = self.mass
        inertia = self.inertia
        arm_length = self.arm_length
        gravity = self.gravity

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        ddx = -(u1 + u2) * sin_theta / mass - gravity * sin_theta
        ddy = (u1 + u2) * cos_theta / mass + gravity * (cos_theta - 1)
        ddtheta = arm_length * (- u1 + u2) / inertia

        return torch.cat([dx, dy, dtheta, ddx, ddy, ddtheta], dim=1)
    
    def linearize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Linearizes the dynamics around the hover position: ẋ = Ax + Bu.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The matrices A and B representing the linearized system.
        """

        mass = self.mass.item()
        inertia = self.inertia.item()
        arm_length = self.arm_length.item()
        gravity = self.gravity.item()

        A = np.zeros((self.state_dim, self.state_dim), dtype=np.float32)
        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0
        A[3, 2] = - gravity

        B = np.zeros((self.state_dim, self.control_dim), dtype=np.float32)
        B[4, 0] = 1.0 / mass
        B[4, 1] = 1.0 / mass
        B[5, 0] = - arm_length / inertia
        B[5, 1] = arm_length / inertia

        return A, B
    
    def get_f_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        mass = self.mass.item()
        inertia = self.inertia.item()
        arm_length = self.arm_length.item()
        gravity = self.gravity.item()

        sin_theta_bound = max_abs_sin(x_lb[2], x_ub[2])
        cos_theta_bound = max_abs_cos(x_lb[2], x_ub[2])
        cos_theta_minus_one_bound = max_abs_cos_minus_one(x_lb[2], x_ub[2])

        f_1_bound = max(abs(x_ub[3]), abs(x_lb[3]))
        f_2_bound = max(abs(x_ub[4]), abs(x_lb[4]))
        f_3_bound = max(abs(x_ub[5]), abs(x_lb[5]))

        tmp1 = (u_lb[0] + u_ub[0]) / mass + gravity
        tmp2 = (u_lb[1] + u_ub[1]) / mass + gravity
        tmp = max(abs(tmp1), abs(tmp2))

        f_4_bound = tmp * sin_theta_bound

        tmp1 = u_lb[0] + u_ub[0]
        tmp2 = u_lb[1] + u_ub[1]
        tmp = max(abs(tmp1), abs(tmp2))
        f_5_bound = tmp / mass * cos_theta_bound + gravity * cos_theta_minus_one_bound

        tmp = max(abs(u_lb[1] - u_lb[0]), abs(u_ub[1] - u_ub[0]), abs(u_lb[1] - u_ub[0]), abs(u_ub[1] - u_lb[0]))
        f_6_bound = arm_length * tmp / inertia

        f_bound = np.sqrt(f_1_bound ** 2 + f_2_bound ** 2 + f_3_bound ** 2 +
                          f_4_bound ** 2 + f_5_bound ** 2 + f_6_bound ** 2)
        return f_bound
    
    def get_f_du_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        mass = self.mass.item()
        inertia = self.inertia.item()
        arm_length = self.arm_length.item()

        sin_theta_bound = max_abs_sin(x_lb[2], x_ub[2])
        cos_theta_bound = max_abs_cos(x_lb[2], x_ub[2])
        
        df_du = np.zeros((self.state_dim, self.control_dim), dtype=np.float32)
        df_du[3, 0] = sin_theta_bound / mass
        df_du[3, 1] = sin_theta_bound / mass
        df_du[4, 0] = cos_theta_bound / mass
        df_du[4, 1] = cos_theta_bound / mass
        df_du[5, 0] = arm_length / inertia
        df_du[5, 1] = arm_length / inertia
            
        df_du_bound = np.linalg.norm(df_du, ord=2)

        return df_du_bound
    
    def get_f_dx_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:
        """Calculates an upper bound on the L2 norm of the Jacobian ∂f/∂x.

        Returns:
            float: Upper bound on the L2 norm of ∂f/∂x.
        """

        mass = self.mass.item()
        gravity = self.gravity.item()

        sin_theta_bound = max_abs_sin(x_lb[2], x_ub[2])
        cos_theta_bound = max_abs_cos(x_lb[2], x_ub[2])

        tmp1 = (u_lb[0] + u_ub[0]) / mass + gravity
        tmp2 = (u_lb[1] + u_ub[1]) / mass + gravity
        tmp = max(abs(tmp1), abs(tmp2))

        df_dx = np.zeros((self.state_dim, self.state_dim), dtype=np.float32)
        df_dx[0, 3] = 1
        df_dx[1, 4] = 1
        df_dx[2, 5] = 1
        df_dx[3, 2] = tmp * cos_theta_bound
        df_dx[4, 2] = tmp * sin_theta_bound

        df_dx_bound = np.linalg.norm(df_dx, ord=2)
        return df_dx_bound
    
    def f_dx(self, x, u):

        theta = x[:, 2]
        u1, u2 = u[:, 0], u[:, 1]

        mass = self.mass.item()
        gravity = self.gravity.item()

        N = x.shape[0]
        f_dx = torch.zeros(N, self.state_dim, self.state_dim, dtype=self.dtype, device=x.device) # (N, 6, 6)
        f_dx[:, 0, 3] = 1.0
        f_dx[:, 1, 4] = 1.0
        f_dx[:, 2, 5] = 1.0
        f_dx[:, 3, 2] = (-gravity - (u1+u2)/mass) * torch.cos(theta)
        f_dx[:, 4, 2] = (-gravity - (u1+u2)/mass) * torch.sin(theta)

        return f_dx
    
    def f_du(self, x, u):

        theta = x[:, 2]
  
        mass = self.mass.item()
        arm_length = self.arm_length.item()
        inertia = self.inertia.item()

        N = x.shape[0]
        f_du = torch.zeros(N, self.state_dim, self.control_dim, dtype=self.dtype, device=x.device)
        f_du[:, 3, 0] = -torch.sin(theta) / mass
        f_du[:, 3, 1] = -torch.sin(theta) / mass
        f_du[:, 4, 0] = torch.cos(theta) / mass
        f_du[:, 4, 1] = torch.cos(theta) / mass
        f_du[:, 5, 0] = -arm_length / inertia
        f_du[:, 5, 1] = arm_length / inertia

        return f_du
    
    def get_f_dxdx_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):

        mass = self.mass.item()
        gravity = self.gravity.item()

        sin_theta_bound = max_abs_sin(x_lb[2], x_ub[2])
        cos_theta_bound = max_abs_cos(x_lb[2], x_ub[2])

        tmp1 = (u_lb[0] + u_ub[0]) / mass + gravity
        tmp2 = (u_lb[1] + u_ub[1]) / mass + gravity
        tmp = float(max(abs(tmp1), abs(tmp2)))

        f_dxdx = torch.zeros(self.state_dim, self.state_dim, self.state_dim, dtype=self.dtype)
        f_dxdx[3,2,2] = tmp * sin_theta_bound
        f_dxdx[4,2,2] = tmp * cos_theta_bound

        f_dxdx_elementwise_l2_bound = torch.linalg.norm(f_dxdx, ord=2, dim=(1,2)) # (6, )

        return f_dxdx_elementwise_l2_bound
    
    def get_f_dxdu_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):

        mass = self.mass.item()

        sin_theta_bound = max_abs_sin(x_lb[2], x_ub[2])
        cos_theta_bound = max_abs_cos(x_lb[2], x_ub[2])

        f_dxdu = torch.zeros(self.state_dim, self.state_dim, self.control_dim, dtype=self.dtype)
        f_dxdu[3,2,0] = cos_theta_bound / mass
        f_dxdu[3,2,1] = cos_theta_bound / mass
        f_dxdu[4,2,0] = sin_theta_bound / mass
        f_dxdu[4,2,1] = sin_theta_bound / mass

        f_dxdu_elementwise_l2_bound = torch.linalg.norm(f_dxdu, ord=2, dim=(1,2)) # (6, )

        return f_dxdu_elementwise_l2_bound
    
    def get_f_dudu_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):
        return torch.zeros(self.state_dim, dtype=self.dtype)