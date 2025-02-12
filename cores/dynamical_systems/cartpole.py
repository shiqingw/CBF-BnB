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

def min_abs_sin(a, b):
    """
    Returns the minimum value of |sin(x)| for x in [a, b].
    """
    # 1. Find integer k such that k*pi could lie in [a, b].
    #    We do this by checking if the set of integers from ceil(a/pi) to floor(b/pi) is non-empty.
    k_min = math.ceil(a / math.pi)
    k_max = math.floor(b / math.pi)
    
    # If there is an integer k in that range, it means k*pi is in [a, b].
    if k_min <= k_max:
        return 0.0  # since sin(k*pi) = 0 for integer k
    
    # 2. Otherwise, return the min of |sin(a)| and |sin(b)|
    return min(abs(math.sin(a)), abs(math.sin(b)))

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

def min_abs_cos(a, b):
    """
    Returns the minimum value of |cos(x)| for x in the closed interval [a, b].
    """
    # 1. Check if there is an integer k such that (pi/2 + k*pi) is in [a, b].
    #    Solve pi/2 + k*pi >= a  and  pi/2 + k*pi <= b
    #    <=> k >= (a - pi/2)/pi  and  k <= (b - pi/2)/pi
    k_min = math.ceil((a - math.pi/2) / math.pi)
    k_max = math.floor((b - math.pi/2) / math.pi)
    
    # If there exists an integer k in that range, cos(x) = 0 for x = pi/2 + k*pi in [a,b].
    if k_min <= k_max:
        return 0.0  # The minimum absolute value is 0
    
    # 2. If no zero of cos(x) is in [a,b], the minimum must be at one of the endpoints.
    return min(abs(math.cos(a)), abs(math.cos(b)))

def max_abs_sin_2x(a, b):
    """
    Returns the maximum value of |sin(2x)| for x in [a, b].
    """
    # 1) Check if there's an integer k such that:
    #       x = pi/4 + (k*pi)/2  lies in [a, b].
    #    Solve:  pi/4 + (k*pi)/2 >= a    and    pi/4 + (k*pi)/2 <= b
    #    <=>     (k*pi)/2 >= a - pi/4    and    (k*pi)/2 <= b - pi/4
    #    <=>     k >= 2*(a - pi/4)/pi    and    k <= 2*(b - pi/4)/pi
    
    k_min = math.ceil( 2*(a - math.pi/4) / math.pi )
    k_max = math.floor(2*(b - math.pi/4) / math.pi )
    
    # If there's an integer k in [k_min, k_max], then sin(2x) = ±1 for some x in [a,b].
    if k_min <= k_max:
        return 1.0
    
    # 2) Otherwise, no point in [a,b] attains ±1, so check the endpoints.
    return max(abs(math.sin(2*a)), abs(math.sin(2*b)))

def max_abs_cos_2x(a, b):
    """
    Returns the maximum value of |cos(2x)| for x in [a, b].
    """
    # 1) We want to find if there's an integer m such that:
    #       x = (m*pi)/2  lies in [a, b].
    #
    #    The condition for x in [a, b] is:
    #       a <= (m*pi)/2 <= b
    #    =>  2a/pi <= m <= 2b/pi
    
    m_min = math.ceil(2 * a / math.pi)
    m_max = math.floor(2 * b / math.pi)
    
    # If there's an integer m in [m_min, m_max], then cos(2x) = ±1 for some x in [a,b].
    if m_min <= m_max:
        return 1.0
    
    # 2) Otherwise, the maximum absolute value must be at one of the endpoints.
    return max(abs(math.cos(2*a)), abs(math.cos(2*b)))

class CartPole(nn.Module):
    def __init__(self, mass_pole: float, mass_cart: float, length: float, friction_coef: float, 
                 dtype: torch.dtype = torch.float32) -> None:
        """Initialize the CartPole system parameters.

        Args:
            mass_pole (float): Mass of the pole.
            mass_cart (float): Mass of the cart.
            length (float): Half the length of the pole (distance from pivot to center of mass).
            friction_coef (float): Coefficient of friction for the cart.
            dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
        """

        super(CartPole, self).__init__()

        self.state_dim = 4
        self.control_dim = 1
        self.dtype = dtype

        self.register_buffer('mass_pole', torch.tensor(mass_pole, dtype=self.dtype))
        self.register_buffer('mass_cart', torch.tensor(mass_cart, dtype=self.dtype))
        self.register_buffer('length', torch.tensor(length, dtype=self.dtype))
        self.register_buffer('friction_coef', torch.tensor(friction_coef, dtype=self.dtype))
        self.register_buffer('gravity', torch.tensor(9.81, dtype=self.dtype))

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Computes the continuous-time dynamics of the cart-pole system.

        Args:
            x (torch.Tensor): State vector [x, θ, ẋ, θ̇], shape = (batch_size, 4)
            u (torch.Tensor): Control input (force), shape = (batch_size, 1)

        Returns:
            torch.Tensor: State derivative [ẋ, θ̇, ẍ, θ̈], shape = (batch_size, 4)
        """

        theta, dx, dtheta = x[:, 1:2], x[:, 2:3], x[:, 3:4]

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        mass_pole = self.mass_pole
        mass_cart = self.mass_cart
        length = self.length
        friction = self.friction_coef
        gravity = self.gravity

        tmp = - length * dtheta**2 + gravity * cos_theta
        ddx = mass_pole * sin_theta * tmp
        ddx += u - torch.sign(dx) * friction * (mass_pole + mass_cart) * gravity 
        ddx /= (mass_cart + mass_pole * sin_theta**2)

        ddtheta = (mass_pole + mass_cart) * gravity * sin_theta
        ddtheta += - mass_pole * length * dtheta**2 * sin_theta * cos_theta
        ddtheta += cos_theta * (u - torch.sign(dx) * friction * (mass_pole + mass_cart) * gravity)
        ddtheta /= (length * (mass_cart + mass_pole * sin_theta**2))

        dstates = torch.cat([dx, dtheta, ddx, ddtheta], dim=1)
        return dstates
    
    def linearize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Linearizes the dynamics around the upright position: ẋ = Ax + Bu.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The matrices A and B representing the linearized system.
        """

        mass_pole = self.mass_pole.item()
        mass_cart = self.mass_cart.item()
        length = self.length.item()
        gravity = self.gravity.item()

        A = np.zeros((self.state_dim, self.state_dim), dtype=np.float32)
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        A[2, 1] = mass_pole * gravity / mass_cart
        A[3, 1] = (mass_pole + mass_cart) * gravity / (length * mass_cart)

        B = np.zeros((self.state_dim, self.control_dim), dtype=np.float32)
        B[2, 0] = 1.0 / mass_cart
        B[3, 0] = 1.0 / (length * mass_cart)

        return A, B
    
    def get_f_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        mass_pole = self.mass_pole.item()
        mass_cart = self.mass_cart.item()
        length = self.length.item()
        friction = self.friction_coef.item()
        gravity = self.gravity.item()
        
        assert friction == 0
        
        f_1 = max(abs(x_lb[2]), abs(x_ub[2]))
        f_2 = max(abs(x_lb[3]), abs(x_ub[3]))

        sin_theta_max = max_abs_sin(x_lb[1], x_ub[1])
        sin_theta_min = min_abs_sin(x_lb[1], x_ub[1])
        cos_theta_max = max_abs_cos(x_lb[1], x_ub[1])
        sin_two_theta_max = max_abs_sin_2x(x_lb[1], x_ub[1])

        dtheta_bound = max(abs(x_lb[3]), abs(x_ub[3]))
        u_bound = max(abs(u_lb[0]), abs(u_ub[0]))

        f_3 = 0.5 * mass_pole * gravity * sin_two_theta_max
        f_3 += mass_pole * length * dtheta_bound**2 * sin_theta_max
        f_3 += u_bound + friction * (mass_pole + mass_cart) * gravity
        f_3 = f_3 / (mass_cart + mass_pole * sin_theta_min**2)

        f_4 = (mass_pole + mass_cart) * gravity * sin_theta_max
        f_4 += 0.5 * mass_pole * length * dtheta_bound**2 * sin_two_theta_max
        f_4 += cos_theta_max * (u_bound + friction * (mass_pole + mass_cart) * gravity)
        f_4 = f_4 / (length * (mass_cart + mass_pole * sin_theta_min**2))

        f_bound = np.sqrt(f_1**2 + f_2**2 + f_3**2 + f_4**2)

        return f_bound.item()
    
    def get_f_du_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:
        """Calculates the L2 norm bound of ∂f/∂u (partial derivative of dynamics w.r.t. control).

        Returns:
            float: Upper bound on the L2 norm of ∂f/∂u.
        """

        mass_cart = self.mass_cart.item()
        mass_pole = self.mass_pole.item()
        length = self.length.item()
        friction = self.friction_coef.item()

        assert friction == 0

        sin_theta_min = min_abs_sin(x_lb[1], x_ub[1])

        df_du = np.zeros((4, 1), dtype=np.float32)
        df_du[2, 0] = 1.0 / (mass_cart + mass_pole * sin_theta_min**2)
        df_du[3, 0] = 1.0 / (length * (mass_cart + mass_pole * sin_theta_min**2))

        return np.linalg.norm(df_du, ord=2)
    
    def get_f_dx_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        mass_pole = self.mass_pole.item()
        mass_cart = self.mass_cart.item()
        length = self.length.item()
        friction = self.friction_coef.item()
        gravity = self.gravity.item()

        assert friction == 0

        sin_theta_max = max_abs_sin(x_lb[1], x_ub[1])
        sin_theta_min = min_abs_sin(x_lb[1], x_ub[1])
        cos_theta_max = max_abs_cos(x_lb[1], x_ub[1])
        sin_two_theta_max = max_abs_sin_2x(x_lb[1], x_ub[1])
        cos_two_theta_max = max_abs_cos_2x(x_lb[1], x_ub[1])
        mc_plus_mp_sin_theta_squared_min = mass_cart + mass_pole * sin_theta_min**2

        dtheta_bound = max(abs(x_lb[3]), abs(x_ub[3]))
        u_bound = max(abs(u_lb[0]), abs(u_ub[0]))

        df_dx = np.zeros((4, 4), dtype=np.float32)
        df_dx[0, 2] = 1.0
        df_dx[1, 3] = 1.0

        tmp1 = mass_pole * gravity * cos_two_theta_max + mass_pole * length * dtheta_bound**2 * cos_theta_max
        tmp1 /= mc_plus_mp_sin_theta_squared_min
        tmp2 = 0.5 * mass_pole * gravity * sin_two_theta_max + mass_pole * length * dtheta_bound**2 * sin_theta_max + u_bound
        tmp2 *= mass_pole * sin_two_theta_max / mc_plus_mp_sin_theta_squared_min**2
        df_dx[2, 1] = tmp1 + tmp2

        df_dx[2, 3] = 2 * mass_pole * length * dtheta_bound * sin_theta_max / mc_plus_mp_sin_theta_squared_min

        tmp3 = (mass_pole + mass_cart) * gravity * cos_theta_max + mass_pole * length * dtheta_bound**2 * cos_two_theta_max + u_bound * sin_theta_max
        tmp3 /= length * mc_plus_mp_sin_theta_squared_min
        tmp4 = (mass_pole + mass_cart) * gravity * sin_theta_max + 0.5 * mass_pole * length * dtheta_bound**2 * sin_two_theta_max + u_bound * cos_theta_max
        tmp4 *= mass_pole * sin_two_theta_max * length / (length * mc_plus_mp_sin_theta_squared_min)**2
        df_dx[3, 1] = tmp3 + tmp4

        df_dx[3, 3] = mass_pole * length * dtheta_bound * sin_two_theta_max / (length * mc_plus_mp_sin_theta_squared_min)

        return np.linalg.norm(df_dx, ord=2)


    def f_dx(self, x, u):

        theta = x[:, 1]
        theta_dot = x[:, 3]

        mass_pole = self.mass_pole.item()
        mass_cart = self.mass_cart.item()
        length = self.length.item()
        friction = self.friction_coef.item()
        gravity = self.gravity.item()

        assert friction == 0

        N = x.shape[0]
        f_dx = torch.zeros(N, self.state_dim, self.state_dim, dtype=self.dtype, device=x.device) # (N, 4, 4)
        f_dx[:, 0, 2] = 1.0
        f_dx[:, 1, 3] = 1.0

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_two_theta = torch.sin(2*theta)
        cos_two_theta = torch.cos(2*theta)
        mc_plus_mp_sin_theta_squared = mass_cart + mass_pole * sin_theta**2

        tmp1 = mass_pole * gravity * cos_two_theta - mass_pole * length * theta_dot**2 * cos_theta
        tmp1 /= mc_plus_mp_sin_theta_squared
        tmp2 = 0.5 * mass_pole * gravity * sin_two_theta - mass_pole * length * theta_dot**2 * sin_theta + u[:, 0]
        tmp2 *= mass_pole * sin_two_theta / mc_plus_mp_sin_theta_squared**2
        f_dx[:,2,1] = tmp1 - tmp2
        f_dx[:,2,3] = - 2 * mass_pole * length * theta_dot * sin_theta / mc_plus_mp_sin_theta_squared

        tmp3 = (mass_pole + mass_cart) * gravity * cos_theta - mass_pole * length * theta_dot**2 * cos_two_theta - u[:, 0] * sin_theta
        tmp3 /= length * mc_plus_mp_sin_theta_squared
        tmp4 = (mass_pole + mass_cart) * gravity * sin_theta - 0.5 * mass_pole * length * theta_dot**2 * sin_two_theta + u[:, 0] * cos_theta
        tmp4 *= mass_pole * sin_two_theta * length / (length * mc_plus_mp_sin_theta_squared)**2
        f_dx[:,3,1] = tmp3 - tmp4

        f_dx[:,3,3] = - mass_pole * length * theta_dot * sin_two_theta / (length * mc_plus_mp_sin_theta_squared)

        return f_dx
    
    def f_du(self, x, u):

        theta = x[:, 1]
        
        mass_cart = self.mass_cart.item()
        mass_pole = self.mass_pole.item()
        length = self.length.item()
        friction = self.friction_coef.item()

        assert friction == 0

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        mc_plus_mp_sin_theta_squared = mass_cart + mass_pole * sin_theta**2

        N = x.shape[0]
        f_du = torch.zeros(N, self.state_dim, self.control_dim, dtype=self.dtype, device=x.device)
        f_du[:, 2, 0] = 1.0 / mc_plus_mp_sin_theta_squared
        f_du[:, 3, 0] = cos_theta / (length * mc_plus_mp_sin_theta_squared)

        return f_du
    
    def get_f_dxdx_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):

        mass_pole = self.mass_pole.item()
        mass_cart = self.mass_cart.item()
        length = self.length.item()
        friction = self.friction_coef.item()
        gravity = self.gravity.item()

        assert friction == 0

        sin_theta_max = max_abs_sin(x_lb[1], x_ub[1])
        sin_theta_min = min_abs_sin(x_lb[1], x_ub[1])
        cos_theta_max = max_abs_cos(x_lb[1], x_ub[1])
        sin_two_theta_max = max_abs_sin_2x(x_lb[1], x_ub[1])
        cos_two_theta_max = max_abs_cos_2x(x_lb[1], x_ub[1])
        mc_plus_mp_sin_theta_squared_min = mass_cart + mass_pole * sin_theta_min**2
        dtheta_bound = max(abs(x_lb[3]), abs(x_ub[3]))
        u_bound = max(abs(u_lb[0]), abs(u_ub[0]))

        f_dxdx = torch.zeros(self.state_dim, self.state_dim, self.state_dim, dtype=self.dtype)

        tmp1 = dtheta_bound**2 * length * mass_pole * sin_theta_max + 0.5 * gravity * mass_pole * sin_two_theta_max + u_bound
        tmp1 *= 2 * mass_pole**2 * sin_two_theta_max**2 / mc_plus_mp_sin_theta_squared_min**3
        tmp2 = dtheta_bound**2 * length * mass_pole * cos_theta_max + gravity * mass_pole * cos_two_theta_max
        tmp2 *= 2 * mass_pole * sin_two_theta_max / mc_plus_mp_sin_theta_squared_min**2
        tmp3 = dtheta_bound**2 * length * mass_pole * sin_theta_max + 0.5 * gravity * mass_pole * sin_two_theta_max + u_bound
        tmp3 *= 2 * mass_pole * cos_two_theta_max**2 / mc_plus_mp_sin_theta_squared_min**2
        tmp4 = dtheta_bound**2 * length * mass_pole * sin_theta_max + 2 * gravity * mass_pole * sin_two_theta_max
        tmp4 /= mc_plus_mp_sin_theta_squared_min
        f_dxdx[2, 1, 1] = float(tmp1 + tmp2 + tmp3 + tmp4)

        tmp5 = 2 * dtheta_bound * length * mass_pole**2 * sin_theta_max * sin_two_theta_max / mc_plus_mp_sin_theta_squared_min**2
        tmp6 = 2 * dtheta_bound * length * mass_pole * cos_theta_max / mc_plus_mp_sin_theta_squared_min
        f_dxdx[2, 1, 3] = float(tmp5 + tmp6)

        f_dxdx[2, 3, 1] = float(tmp5 + tmp6)

        f_dxdx[2, 3, 3] = float(2 * length * mass_pole * sin_theta_max / mc_plus_mp_sin_theta_squared_min)

        tmp7 = 0.5 * dtheta_bound**2 * length * mass_pole * sin_two_theta_max 
        tmp7 += gravity * (mass_cart + mass_pole) * sin_theta_max + u_bound * cos_theta_max
        tmp7 *= 2 * mass_pole**2 * sin_two_theta_max**2 / (length * mc_plus_mp_sin_theta_squared_min**3 )
        tmp8 = 0.5 * dtheta_bound**2 * length * mass_pole * cos_two_theta_max 
        tmp8 += gravity * (mass_cart + mass_pole) * sin_theta_max + u_bound * cos_theta_max
        tmp8 *= 2 * mass_pole * cos_two_theta_max / (length * mc_plus_mp_sin_theta_squared_min**2)
        tmp9 = dtheta_bound**2 * length * mass_pole * cos_two_theta_max 
        tmp9 += gravity * (mass_cart + mass_pole) * cos_theta_max + u_bound * sin_theta_max
        tmp9 *= 2 * mass_pole * sin_two_theta_max / (length * mc_plus_mp_sin_theta_squared_min**2)
        tmp10 = 2.0 * dtheta_bound**2 * length * mass_pole * sin_two_theta_max 
        tmp10 += gravity * (mass_cart + mass_pole) * sin_theta_max + u_bound * cos_theta_max
        tmp10 /= length * mc_plus_mp_sin_theta_squared_min
        f_dxdx[3, 1, 1] = float(tmp7 + tmp8 + tmp9 + tmp10)

        tmp11 = dtheta_bound * mass_pole**2 * sin_two_theta_max**2 / mc_plus_mp_sin_theta_squared_min**2
        tmp12 = 2 * dtheta_bound * mass_pole * cos_two_theta_max / mc_plus_mp_sin_theta_squared_min
        f_dxdx[3, 1, 3] = float(tmp11 + tmp12)

        f_dxdx[3, 3, 1] = float(tmp11 + tmp12)

        f_dxdx[3, 3, 3] = float(mass_pole * sin_two_theta_max / mc_plus_mp_sin_theta_squared_min)

        f_dxdx_elementwise_l2_bound = torch.linalg.norm(f_dxdx, ord=2, dim=(1,2)) # (2,)
        return f_dxdx_elementwise_l2_bound

    def get_f_dxdu_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):

        mass_pole = self.mass_pole.item()
        mass_cart = self.mass_cart.item()
        length = self.length.item()
        friction = self.friction_coef.item()

        assert friction == 0

        sin_theta_max = max_abs_sin(x_lb[1], x_ub[1])
        sin_theta_min = min_abs_sin(x_lb[1], x_ub[1])
        cos_theta_max = max_abs_cos(x_lb[1], x_ub[1])
        sin_two_theta_max = max_abs_sin_2x(x_lb[1], x_ub[1])
        mc_plus_mp_sin_theta_squared_min = mass_cart + mass_pole * sin_theta_min**2

        f_dxdu = torch.zeros(self.state_dim, self.state_dim, self.control_dim, dtype=self.dtype)

        f_dxdu[2, 1, 0] = mass_pole * sin_two_theta_max / mc_plus_mp_sin_theta_squared_min**2

        tmp1 = mass_pole * cos_theta_max * sin_two_theta_max / (length * mc_plus_mp_sin_theta_squared_min**2)
        tmp2 = sin_theta_max / (length * mc_plus_mp_sin_theta_squared_min)
        f_dxdu[3, 1, 0] = tmp1 + tmp2

        return torch.linalg.norm(f_dxdu, ord=2, dim=(1,2)) 
    
    def get_f_dudu_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):
        
        return torch.zeros(self.state_dim, dtype=self.dtype)