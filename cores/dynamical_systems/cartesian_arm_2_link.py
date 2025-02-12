import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union

class CartesianArmTwoLink(nn.Module):
    def __init__(self, mass_link_1: float, mass_link_2: float, dtype: torch.dtype = torch.float32) -> None:
        super(CartesianArmTwoLink, self).__init__()

        self.state_dim = 4
        self.control_dim = 2
        self.dtype = dtype

        self.register_buffer('mass_link_1', torch.tensor(mass_link_1, dtype=self.dtype))
        self.register_buffer('mass_link_2', torch.tensor(mass_link_2, dtype=self.dtype))

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.state_dim, "Invalid state dimension."
        assert u.shape[1] == self.control_dim, "Invalid control dimension."

        dq1, dq2 = x[:, 2:3], x[:, 3:4]
        tau1, tau2 = u[:, 0:1], u[:, 1:2]

        m_link_1 = self.mass_link_1
        m_link_2 = self.mass_link_2
        ddq1 = tau1 / (m_link_1 + m_link_2)
        ddq2 = tau2 / m_link_2

        return torch.cat([dq1, dq2, ddq1, ddq2], dim=1)
    
    def linearize(self) -> Tuple[np.ndarray, np.ndarray]:

        m_link_1 = self.mass_link_1
        m_link_2 = self.mass_link_2

        A = np.zeros((self.state_dim, self.state_dim), dtype=np.float32)
        A[0, 2] = 1.0
        A[1, 3] = 1.0

        B = np.zeros((self.state_dim, self.control_dim), dtype=np.float32)
        B[2, 0] = 1.0 / (m_link_1 + m_link_2)
        B[3, 1] = 1.0 / m_link_2

        return A, B
    
    def get_f_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:
        m_link_1 = self.mass_link_1.item()
        m_link_2 = self.mass_link_2.item()

        f_1_bound = max(abs(x_lb[2]), abs(x_ub[2]))
        f_2_bound = max(abs(x_lb[3]), abs(x_ub[3]))
        f_3_bound = max(abs(u_lb[0]), abs(u_ub[0])) / (m_link_1 + m_link_2)
        f_4_bound = max(abs(u_lb[1]), abs(u_ub[1])) / m_link_2

        f_bound = np.sqrt(f_1_bound ** 2 + f_2_bound ** 2 + f_3_bound ** 2 + f_4_bound ** 2)

        return f_bound
    
    def get_f_du_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        m_link_1 = self.mass_link_1.item()
        m_link_2 = self.mass_link_2.item()

        df_du = np.zeros((self.state_dim, self.control_dim), dtype=np.float32)
        df_du[2, 0] = 1.0 / (m_link_1 + m_link_2)
        df_du[3, 1] = 1.0 / m_link_2

        df_du_bound = np.linalg.norm(df_du, ord=2)
        return df_du_bound
    
    def get_f_dx_l2_bound(self, x_lb, x_ub, u_lb, u_ub) -> float:

        df_dx = np.zeros((self.state_dim, self.state_dim), dtype=np.float32)
        df_dx[0, 2] = 1.0
        df_dx[1, 3] = 1.0
        
        df_dx_bound = np.linalg.norm(df_dx, ord=2)
        return df_dx_bound

    def f_dx(self, x, u):

        N = x.shape[0]
        f_dx = torch.zeros(N, self.state_dim, self.state_dim, dtype=self.dtype, device=x.device) # (N, 4, 4)
        f_dx[:, 0, 2] = 1.0
        f_dx[:, 1, 3] = 1.0

        return f_dx
    
    def f_du(self, x, u):

        m_link_1 = self.mass_link_1.item()
        m_link_2 = self.mass_link_2.item()

        N = x.shape[0]
        f_du = torch.zeros(N, self.state_dim, self.control_dim, dtype=self.dtype, device=x.device)
        f_du[:, 2, 0] = 1.0 / (m_link_1 + m_link_2)
        f_du[:, 3, 1] = 1.0 / m_link_2

        return f_du
    
    def get_f_dxdx_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):

        return torch.zeros(self.state_dim, dtype=self.dtype)

    def get_f_dxdu_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):

        return torch.zeros(self.state_dim, dtype=self.dtype)
    
    def get_f_dudu_elementwise_l2_bound(self, x_lb, x_ub, u_lb, u_ub):
        
        return torch.zeros(self.state_dim, dtype=self.dtype)
