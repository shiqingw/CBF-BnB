import torch
from torch.utils.data import TensorDataset, DataLoader

from .adaptive_mesh import AdaptiveMesh

class FeasibilityAdaptiveMesh(AdaptiveMesh):
    def __init__(self, cbf_nn, lip_h_l2, hess_h_l2, thrid_order_h_elementwise_l2_bound, 
                 control_lower_bound, control_upper_bound,
                 system, F_l1_bound, F_dx_l2_bound, F_dxdx_elementwise_l2_bound,
                 disturbance_elementwise_upper_bound, disturbance_channel, 
                 cbf_alpha, lip_zero_order_l2,
                 state_dim, x_lb, x_ub, batch_size, dtype, device):
        super().__init__(x_lb, x_ub, dtype)
        self.cbf_nn = cbf_nn # h(x)
        self.lip_h_l2 = lip_h_l2
        self.hess_h_l2 = hess_h_l2
        self.h_l2_elementwise_third_order_bound = thrid_order_h_elementwise_l2_bound

        self.control_lower_bound = control_lower_bound 
        self.control_upper_bound = control_upper_bound

        self.system = system # F(x, u)
        self.F_l1_bound = F_l1_bound # scalar
        self.F_dx_l2_bound = F_dx_l2_bound # scalar
        assert F_dxdx_elementwise_l2_bound.dim() == 1, "Invalid F_dxdx_l2_bound dimension."
        assert len(F_dxdx_elementwise_l2_bound) == state_dim, "Invalid F_dxdx_l2_bound dimension."
        self.F_dxdx_l2_bound = torch.linalg.norm(F_dxdx_elementwise_l2_bound, ord=2) # scalar
        
        self.disturbance_elementwise_upper_bound = disturbance_elementwise_upper_bound
        self.disturbance_channel = disturbance_channel # G, shape (n, n_d)
        self.disturbance_channel_columnwise_l1 = torch.linalg.norm(disturbance_channel, ord=1, dim=0) # shape (n_d,)

        self.cbf_alpha = cbf_alpha
        self.lip_zero_order_l2 = lip_zero_order_l2
        self.state_dim = state_dim

        self.batch_size = batch_size
        self.device = device

        self.mesh_radius_l2 = torch.linalg.norm(x_ub - x_lb, ord=2)/2.0
    
    def bound(self):
        data_loader = DataLoader(self.regions, batch_size=self.batch_size, shuffle=False)
        psi_lb = torch.zeros(len(self.regions), dtype=self.dtype)
        psi_ub = torch.zeros(len(self.regions), dtype=self.dtype)
        for i, (x_lb_batch, x_ub_batch) in enumerate(data_loader):
            x_lb_batch = x_lb_batch.to(self.device)
            x_ub_batch = x_ub_batch.to(self.device)
            x_m_batch = (x_lb_batch + x_ub_batch) / 2.0 # shape (N, n)
            c_vec_batch = (x_ub_batch - x_lb_batch) / 2.0 # shape (N, n)

            # Value, gradient, and Hessian of h
            h_batch, h_dx_batch, h_dxdx_batch = self.cbf_nn.forward_with_jacobian_and_hessian_method2(x_m_batch) # shape (N,1), (N, 1, n), (N, 1, n, n)
            h_batch = h_batch.squeeze(1) # shape (N,)
            h_dx_batch = h_dx_batch.squeeze(1) # shape (N, n)
            h_dxdx_batch = h_dxdx_batch.squeeze(1) # shape (N, n, n)

            g_batch = self.system.get_actuation(x_m_batch) # shape (N, n, m)
            h_dx_g_batch = torch.bmm(h_dx_batch.unsqueeze(1), g_batch).squeeze(1) # (safe_size, control_dim)
            u_batch = torch.where(h_dx_g_batch >= 0, self.control_upper_bound.unsqueeze(0), self.control_lower_bound.unsqueeze(0))

            F_batch = self.system(x_m_batch, u_batch) # shape (N, n)
            f_dx_batch = self.system.f_dx(x_m_batch, u_batch) # shape (N, n, n)

            # psi value
            psi_batch = (h_dx_batch * F_batch).sum(dim=1) # shape (N,)
            h_dx_G_batch = torch.matmul(h_dx_batch, self.disturbance_channel) # shape (N, n_d)
            abs_h_dx_G_batch = torch.abs(h_dx_G_batch) # shape (N, n_d)
            psi_batch += self.cbf_alpha * h_batch # shape (N,)
            psi_batch += -(abs_h_dx_G_batch * self.disturbance_elementwise_upper_bound).sum(dim=1) # shape (N,)

            # Zero order bound
            ## Lower bound
            psi_lb_batch_zero_order = psi_batch - self.lip_zero_order_l2 * self.mesh_radius_l2 # shape (N,)
            ## Upper bound
            psi_ub_batch_zero_order = psi_batch.clone() # shape (N,)

            # First-order bound
            ## Lower bound
            psi_lb_batch_first_order = psi_batch.clone() # shape (N,)
            ### Disturbance part
            h_dxdx_G_batch = torch.matmul(h_dxdx_batch, self.disturbance_channel) # shape (N, n, n_d)
            abs_h_dxdx_G_batch = torch.abs(h_dxdx_G_batch) # shape (N, n, n_d)
            psi_lb_batch_first_order += -(torch.bmm(c_vec_batch.unsqueeze(1), abs_h_dxdx_G_batch).squeeze(1) *\
                                          self.disturbance_elementwise_upper_bound).sum(dim=1) # shape (N,)
            psi_lb_batch_first_order += -0.5 * self.h_l2_elementwise_third_order_bound * self.mesh_radius_l2**2 *\
                torch.sum(self.disturbance_channel_columnwise_l1 * self.disturbance_elementwise_upper_bound) # shape (N,)
            ### Dynamics part
            first_order_coef = torch.bmm(F_batch.unsqueeze(1), h_dxdx_batch).squeeze(1) # shape (N, n)
            first_order_coef += torch.bmm(h_dx_batch.unsqueeze(1), f_dx_batch).squeeze(1) # shape (N, n)
            first_order_coef += self.cbf_alpha * h_dx_batch # shape (N, n)
            abs_first_order_coef = torch.abs(first_order_coef) # shape (N, n)
            psi_lb_batch_first_order += - (abs_first_order_coef * c_vec_batch).sum(dim=1) # shape (N,)

            second_order_coef = 2 * self.F_dx_l2_bound * self.hess_h_l2
            second_order_coef += self.F_l1_bound * self.h_l2_elementwise_third_order_bound
            second_order_coef += self.lip_h_l2 * self.F_dxdx_l2_bound
            second_order_coef += self.cbf_alpha * self.hess_h_l2
            psi_lb_batch_first_order += -0.5 * second_order_coef * self.mesh_radius_l2**2

            ## Upper bound
            psi_ub_batch_first_order = psi_batch.clone() # shape (N,)
            psi_ub_batch_first_order += - (abs_first_order_coef * c_vec_batch).sum(dim=1) # shape (N,)
            psi_ub_batch_first_order += 0.5 * second_order_coef * self.mesh_radius_l2**2

            # Keep the least conservative one
            psi_lb_batch = torch.max(psi_lb_batch_zero_order, psi_lb_batch_first_order)
            psi_ub_batch = torch.min(psi_ub_batch_zero_order, psi_ub_batch_first_order)

            # Set psi_ub_batch to inf if x_m safisties the constraint: 
            # 1. h(x_m) < 0 and psi_ub(x_m) < 0
            psi_ub_batch[torch.where((h_batch < 0) & (psi_ub_batch < 0))] = float("inf")

            # Copy to f_lb and f_ub
            psi_lb[i*self.batch_size:min((i+1)*self.batch_size,len(self.regions))] = psi_lb_batch.detach().cpu()
            psi_ub[i*self.batch_size:min((i+1)*self.batch_size,len(self.regions))] = psi_ub_batch.detach().cpu()

        return psi_lb, psi_ub
    
    def refine(self, refine_idx):
        x_lb, x_ub = self.regions[refine_idx]
        x_lb_new, x_ub_new = self.split(x_lb, x_ub)
        xi_max = torch.max(torch.abs(x_lb_new), torch.abs(x_ub_new))
        xi_max_l2 = torch.linalg.norm(xi_max, ord=2, dim=1) # shape (N,)
        self.mesh_radius_l2 = torch.linalg.norm(x_ub_new[0] - x_lb_new[0], ord=2)/2.0

        N = len(x_lb_new)
        h_ub = torch.zeros(N, dtype=self.dtype)
        data_set = TensorDataset(x_lb_new, x_ub_new)
        data_loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=False)
        for i, (x_lb_batch, x_ub_batch) in enumerate(data_loader):
            x_lb_batch = x_lb_batch.to(self.device)
            x_ub_batch = x_ub_batch.to(self.device)
            x_m_batch = (x_lb_batch + x_ub_batch) / 2.0 # shape (N, n)
            c_vec_batch = (x_ub_batch - x_lb_batch) / 2.0 # shape (N, n)

            h_batch, h_dx_batch = self.cbf_nn.forward_with_jacobian(x_m_batch) # shape (N, 1), (N, 1 ,n)
            h_batch = h_batch.squeeze(1) # shape (N,)
            h_dx_batch = h_dx_batch.squeeze(1) # shape (N, n)

            # Bound h value (minimization)
            h_ub_batch_zero_order = h_batch + self.lip_h_l2 * self.mesh_radius_l2 # shape (N,)
            h_ub_batch_first_order = h_batch.clone() # shape (N,)
            h_ub_batch_first_order += torch.sum(torch.abs(h_dx_batch) * c_vec_batch, dim=1) # shape (N,)
            h_ub_batch_first_order += 0.5 * self.hess_h_l2 * self.mesh_radius_l2**2 # shape (N,)

            h_ub_batch = torch.min(h_ub_batch_zero_order, h_ub_batch_first_order) # shape (N,), keep the least conservative one

            h_ub[i*self.batch_size:min((i+1)*self.batch_size,N)] = h_ub_batch

        keep_idx = torch.where(h_ub > 0)[0]
        self.regions = TensorDataset(x_lb_new[keep_idx], 
                                     x_ub_new[keep_idx])
        return

