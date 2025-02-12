import torch
from torch.utils.data import TensorDataset, DataLoader

from .adaptive_mesh import AdaptiveMesh

class StabilityProbAdaptiveMeshFirstOrder(AdaptiveMesh):
    def __init__(self, V, lip_V_l2, hess_V_l2, thrid_order_V_elementwise_bound_l2, 
                 controller, lip_controller_l2,
                 system, f_dx_l2_bound, f_du_l2_bound, f_dxdx_l2_bound, f_dxdu_l2_bound, f_dudu_l2_bound, 
                 disturbance_bound, disturbance_channel, lip_disturbance_bound_l2, 
                 beta, lip_zero_order_l2,
                 state_dim, x_lb, x_ub, batch_size, cutoff_radius, V_c,
                 dtype, device):
        super().__init__(x_lb, x_ub, dtype)
        self.V = V # V(x)
        self.lip_V_l2 = lip_V_l2
        self.hess_V_l2 = hess_V_l2
        self.lyapunov_l2_elementwise_third_order_bound = thrid_order_V_elementwise_bound_l2

        self.controller = controller # k(x)
        self.lip_controller_l2 = lip_controller_l2

        self.system = system # f(x, k(x))
        self.f_dx_l2_bound = f_dx_l2_bound # scalar
        self.f_du_l2_bound = f_du_l2_bound # scalar
        self.f_dxdx_l2_bound = f_dxdx_l2_bound # shape (n,)
        assert self.f_dxdx_l2_bound.dim() == 1, "Invalid f_dxdx_l2_bound dimension."
        assert len(self.f_dxdx_l2_bound) == state_dim, "Invalid f_dxdx_l2_bound dimension."
        self.f_dxdu_l2_bound = f_dxdu_l2_bound # shape (n,)
        assert len(self.f_dxdu_l2_bound) == state_dim, "Invalid f_dxdu_l2_bound dimension."
        assert len(self.f_dxdu_l2_bound) == state_dim, "Invalid f_dxdu_l2_bound dimension."
        self.f_dudu_l2_bound = f_dudu_l2_bound # shape (n,)
        assert self.f_dudu_l2_bound.dim() == 1, "Invalid f_dudu_l2_bound dimension."
        assert len(self.f_dudu_l2_bound) == state_dim, "Invalid f_dudu_l2_bound dimension."

        self.disturbance_bound = disturbance_bound # \epsilon(x)
        self.disturbance_channel = disturbance_channel # G, shape (n, n_d)
        self.lip_disturbance_bound_l2 = lip_disturbance_bound_l2
        self.norm_disturbance_channel_l2 = torch.linalg.norm(disturbance_channel, ord=2)

        self.beta = beta
        self.lip_zero_order_l2 = lip_zero_order_l2
        self.state_dim = state_dim

        self.batch_size = batch_size
        self.cutoff_radius = cutoff_radius
        self.V_c = V_c
        self.device = device

        self.mesh_radius_l2 = torch.linalg.norm(x_ub - x_lb, ord=2)/2.0
        self.hess_f_elementwise_l2 = (self.f_dxdx_l2_bound + 2 * self.f_dxdu_l2_bound * self.lip_controller_l2 +\
            self.f_dudu_l2_bound * self.lip_controller_l2**2).to(device) # shape (n,)
        
        assert self.hess_f_elementwise_l2.dim() == 1, "Invalid hess_f_elementwise_l2 dimension."
        assert len(self.hess_f_elementwise_l2) == self.state_dim, "Invalid hess_f_elementwise_l2 dimension."
    
    def bound(self):
        data_loader = DataLoader(self.regions, batch_size=self.batch_size, shuffle=False)
        psi_lb = torch.zeros(len(self.regions), dtype=self.dtype)
        psi_ub = torch.zeros(len(self.regions), dtype=self.dtype)
        for i, (x_lb_batch, x_ub_batch) in enumerate(data_loader):
            x_lb_batch = x_lb_batch.to(self.device)
            x_ub_batch = x_ub_batch.to(self.device)
            x_m_batch = (x_lb_batch + x_ub_batch) / 2.0 # shape (N, n)
            c_vec_batch = (x_ub_batch - x_lb_batch) / 2.0 # shape (N, n)

            # Value, gradient, and Hessian of V
            V_batch, V_dx_batch, V_dxdx_batch = self.V.forward_with_jacobian_and_hessian_method2(x_m_batch) # shape (N,1), (N, 1, n), (N, 1, n, n)
            V_batch = V_batch.squeeze(1) # shape (N,)
            V_dx_batch = V_dx_batch.squeeze(1) # shape (N, n)
            V_dxdx_batch = V_dxdx_batch.squeeze(1) # shape (N, n, n)

            # Closed-loop dynamics
            u_batch = self.controller(x_m_batch) # shape (N, m)
            f_batch = self.system(x_m_batch, u_batch) # shape (N, n)
            f_dx_batch = self.system.f_dx(x_m_batch, u_batch) # shape (N, n, n)
            f_du_batch = self.system.f_du(x_m_batch, u_batch) # shape (N, n, m)

            # psi value
            dV_batch = (V_dx_batch * f_batch).sum(dim=1) # shape (N,)
            V_dx_G_batch = torch.matmul(V_dx_batch, self.disturbance_channel) # shape (N, n_d)
            norm_V_dx_G_l2_batch = torch.linalg.norm(V_dx_G_batch, ord=2, dim=1) # shape (N,)
            epsilon_batch = self.disturbance_bound(x_m_batch) # shape (N,)
            norm_xm_batch = torch.linalg.norm(x_m_batch, ord=2, dim=1) # shape (N,)
            psi_batch = dV_batch + norm_V_dx_G_l2_batch * epsilon_batch + self.beta * norm_xm_batch

            # Zero order bound
            ## Lower bound
            psi_lb_batch_zero_order = psi_batch.clone() # shape (N,)
            ## Upper bound
            psi_ub_batch_zero_order = psi_batch + self.lip_zero_order_l2 * self.mesh_radius_l2 # shape (N,)

            # Disturbance part (common to mixed zero order bound and mixed first order bound)
            ## First order terms
            dist_delta_first_order_coefs_batch = torch.zeros(len(x_m_batch), dtype=self.dtype, device=self.device) # shape (N,)
            ### Part not included in dist_delta_first_order_coefs_batch
            V_dxdx_G_batch = torch.matmul(V_dxdx_batch, self.disturbance_channel) # shape (N, n, n_d)
            abs_V_dxdx_G_batch = torch.abs(V_dxdx_G_batch) # shape (N, n, n_d)
            abs_V_dxdx_G_transpose_batch= abs_V_dxdx_G_batch.transpose(1,2) # shape (N, n_d, n)
            abs_V_dxdx_G_batch_transpose_c_vec_batch = torch.bmm(abs_V_dxdx_G_transpose_batch, c_vec_batch.unsqueeze(-1)).squeeze(-1) # shape (N, n_d)
            dist_delta_first_order_not_included_batch = torch.linalg.norm(abs_V_dxdx_G_batch_transpose_c_vec_batch, ord=2, dim=1) * epsilon_batch # shape (N,)

            ### Part included in dist_delta_first_order_coefs_batch
            norm_V_dxdx_G_l2_batch = torch.linalg.norm(V_dxdx_G_batch, ord=2, dim=(1,2)) # shape (N,)
            dist_delta_first_order_coefs_batch += norm_V_dx_G_l2_batch * self.lip_disturbance_bound_l2 # shape (N,)

            ## Second order terms
            dist_delta_second_order_coefs_batch = torch.zeros(len(x_m_batch), dtype=self.dtype, device=self.device) # shape (N,)
            dist_delta_second_order_coefs_batch += 0.5 * self.state_dim**0.5 * self.lyapunov_l2_elementwise_third_order_bound * \
                self.norm_disturbance_channel_l2 * epsilon_batch # shape (N,)
            dist_delta_second_order_coefs_batch += norm_V_dxdx_G_l2_batch * self.lip_disturbance_bound_l2 # shape (N,)

            ## Third order terms
            dist_delta_third_order_coefs_batch = torch.zeros(len(x_m_batch), dtype=self.dtype, device=self.device) # shape (N,)
            dist_delta_third_order_coefs_batch += 0.5 * self.state_dim**0.5 * self.lyapunov_l2_elementwise_third_order_bound * \
                self.norm_disturbance_channel_l2 * self.lip_disturbance_bound_l2 # shape (N,)

            # Positive definite part (common to mixed zero order bound and mixed first order bound)
            pos_def_delta_first_order_coefs_batch = self.beta # scalar

            # Mixed zero order bound
            ## Lower bound
            psi_lb_batch_mixed_zero_order = psi_batch.clone() # shape (N,)
            ## Upper bound
            psi_ub_batch_mixed_zero_order = psi_batch.clone() # shape (N,)

            norm_V_dx_l2_batch = torch.linalg.norm(V_dx_batch, ord=2, dim=1) # shape (N,)
            psi_ub_batch_mixed_zero_order += norm_V_dx_l2_batch * self.f_dx_l2_bound * self.mesh_radius_l2 # shape (N,)
            psi_ub_batch_mixed_zero_order += norm_V_dx_l2_batch * self.f_du_l2_bound * self.lip_controller_l2 * self.mesh_radius_l2 # shape (N,)

            norm_f_l2_batch = torch.linalg.norm(f_batch, ord=2, dim=1) # shape (N,)
            psi_ub_batch_mixed_zero_order += self.hess_V_l2 * norm_f_l2_batch * self.mesh_radius_l2 # shape (N,)

            psi_ub_batch_mixed_zero_order += self.hess_V_l2 * self.f_dx_l2_bound * self.mesh_radius_l2**2 # shape (N,)

            psi_ub_batch_mixed_zero_order += self.hess_V_l2 * self.f_du_l2_bound * self.lip_controller_l2 * self.mesh_radius_l2**2 # shape (N,)

            psi_ub_batch_mixed_zero_order += dist_delta_first_order_not_included_batch # shape (N,)
            psi_ub_batch_mixed_zero_order += dist_delta_first_order_coefs_batch * self.mesh_radius_l2 # shape (N,)
            psi_ub_batch_mixed_zero_order += dist_delta_second_order_coefs_batch * self.mesh_radius_l2**2 # shape (N,)
            psi_ub_batch_mixed_zero_order += dist_delta_third_order_coefs_batch * self.mesh_radius_l2**3 # shape (N,)

            psi_ub_batch_mixed_zero_order += pos_def_delta_first_order_coefs_batch * self.mesh_radius_l2 # shape (N,)

            # Mixed first order bound
            ## Upper bound
            psi_ub_batch_mixed_first_order = psi_batch.clone() # shape (N,)
            ### first order terms
            #### Part not included in mfo_first_order_coefs_batch
            V_dx_f_dx_batch = torch.bmm(V_dx_batch.unsqueeze(1), f_dx_batch) # shape (N, 1, n)
            V_dx_f_dx_batch = V_dx_f_dx_batch.squeeze(1) # shape (N, n)

            V_dxdx_f_batch = torch.bmm(V_dxdx_batch, f_batch.unsqueeze(-1)) # shape (N, n, 1)
            V_dxdx_f_batch = V_dxdx_f_batch.squeeze(-1) # shape (N, n)

            V_dx_f_dx_plus_V_dxdx_f_batch = V_dx_f_dx_batch + V_dxdx_f_batch # shape (N, n)
            abs_V_dx_f_dx_plus_V_dxdx_f_batch = torch.abs(V_dx_f_dx_plus_V_dxdx_f_batch) # shape (N, n)
            psi_ub_batch_mixed_first_order += torch.sum(abs_V_dx_f_dx_plus_V_dxdx_f_batch * c_vec_batch, dim=1) # shape (N,)

            #### Part included in mfo_first_order_coefs_batch
            V_dx_f_du_batch = torch.bmm(V_dx_batch.unsqueeze(1), f_du_batch) # shape (N, 1, m)
            V_dx_f_du_batch = V_dx_f_du_batch.squeeze(1) # shape (N, m)
            mfo_first_order_coefs_batch = torch.linalg.norm(V_dx_f_du_batch, ord=2, dim=1) * self.lip_controller_l2 
            psi_ub_batch_mixed_first_order += mfo_first_order_coefs_batch * self.mesh_radius_l2 # shape (N,)

            ### second order terms
            mfo_second_order_coefs_batch = torch.zeros(len(x_m_batch), dtype=self.dtype, device=self.device) # shape (N,)
            abs_V_dx_batch = torch.abs(V_dx_batch) # shape (N, n)
            abs_V_dx_hess_f_elementwise_batch = torch.mv(abs_V_dx_batch, self.hess_f_elementwise_l2) # shape (N,)
            mfo_second_order_coefs_batch += 0.5 * abs_V_dx_hess_f_elementwise_batch # shape (N,)

            V_dxdx_f_dx_batch = torch.bmm(V_dxdx_batch, f_dx_batch) # shape (N, n, n)
            mfo_second_order_coefs_batch += torch.linalg.norm(V_dxdx_f_dx_batch, ord=2, dim=(1,2)) # shape (N,)

            V_dxdx_f_du_batch = torch.bmm(V_dxdx_batch, f_du_batch) # shape (N, n, m)
            mfo_second_order_coefs_batch += torch.linalg.norm(V_dxdx_f_du_batch, ord=2, dim=(1,2)) * self.lip_controller_l2 # shape (N,)

            mfo_second_order_coefs_batch += 0.5 * self.lyapunov_l2_elementwise_third_order_bound * torch.linalg.norm(f_batch, ord=1, dim=1) # shape (N,)
            psi_ub_batch_mixed_first_order += mfo_second_order_coefs_batch * self.mesh_radius_l2**2 # shape (N,)

            ### Third order terms
            mfo_third_order_coefs_batch = torch.zeros(len(x_m_batch), dtype=self.dtype, device=self.device) # shape (N,)

            abs_V_dxdx_batch = torch.abs(V_dxdx_batch) # shape (N, n, n)
            abs_V_dxdx_batch_hess_f_elementwise_batch = torch.matmul(abs_V_dxdx_batch, self.hess_f_elementwise_l2) # shape (N, n)
            mfo_third_order_coefs_batch += 0.5 * torch.linalg.norm(abs_V_dxdx_batch_hess_f_elementwise_batch, ord=2, dim=1) # shape (N,)

            abs_f_dx = torch.abs(f_dx_batch) # shape (N, n, n)
            abs_f_dx_column_sum = torch.sum(abs_f_dx, dim=1) # shape (N, n)
            mfo_third_order_coefs_batch += 0.5 * self.lyapunov_l2_elementwise_third_order_bound * torch.linalg.norm(abs_f_dx_column_sum, ord=2, dim=1) # shape (N,)

            abs_f_du = torch.abs(f_du_batch) # shape (N, n, m)
            abs_f_du_column_sum = torch.sum(abs_f_du, dim=1) # shape (N, m)
            mfo_third_order_coefs_batch += 0.5 * self.lyapunov_l2_elementwise_third_order_bound * torch.linalg.norm(abs_f_du_column_sum, ord=2, dim=1) * \
                self.lip_controller_l2  # shape (N,)
            
            psi_ub_batch_mixed_first_order += mfo_third_order_coefs_batch * self.mesh_radius_l2**3 # shape (N,)

            ### Fourth order terms
            mfo_fourth_order_coefs_batch = torch.zeros(len(x_m_batch), dtype=self.dtype, device=self.device) # shape (N,)
            mfo_fourth_order_coefs_batch += 0.25 * self.lyapunov_l2_elementwise_third_order_bound * torch.linalg.norm(self.hess_f_elementwise_l2, ord=1) # shape (N,)

            psi_ub_batch_mixed_first_order += mfo_fourth_order_coefs_batch * self.mesh_radius_l2**4 # shape (N,)

            psi_ub_batch_mixed_first_order += dist_delta_first_order_not_included_batch # shape (N,)
            psi_ub_batch_mixed_first_order += dist_delta_first_order_coefs_batch * self.mesh_radius_l2 # shape (N,)
            psi_ub_batch_mixed_first_order += dist_delta_second_order_coefs_batch * self.mesh_radius_l2**2 # shape (N,)
            psi_ub_batch_mixed_first_order += dist_delta_third_order_coefs_batch * self.mesh_radius_l2**3 # shape (N,)

            psi_ub_batch_mixed_first_order += pos_def_delta_first_order_coefs_batch * self.mesh_radius_l2 # shape (N,)

            ## Lower bound
            psi_lb_batch_mixed_first_order = psi_batch.clone() # shape (N,)

            # Keep the least conservative one
            psi_lb_batch = torch.max(psi_lb_batch_zero_order, torch.max(psi_lb_batch_mixed_zero_order, psi_lb_batch_mixed_first_order))
            psi_ub_batch = torch.min(psi_ub_batch_zero_order, torch.min(psi_ub_batch_mixed_zero_order, psi_ub_batch_mixed_first_order))

            # Set psi_lb_batch to -inf if x_m id: 
            # 1. inside the cutoff radius and positive
            psi_lb_batch[torch.where((norm_xm_batch <= self.cutoff_radius) & (psi_lb_batch > 0))] = -float("inf")

            # Copy to f_lb and f_ub
            psi_lb[i*self.batch_size:min((i+1)*self.batch_size,len(self.regions))] = psi_lb_batch.detach().cpu()
            psi_ub[i*self.batch_size:min((i+1)*self.batch_size,len(self.regions))] = psi_ub_batch.detach().cpu()

        return psi_lb, psi_ub
    
    def refine(self, refine_idx):
        x_lb, x_ub = self.regions[refine_idx]
        x_lb_new, x_ub_new = self.split(x_lb, x_ub)
        x_m_new = (x_lb_new + x_ub_new) / 2.0
        xi_max = torch.max(torch.abs(x_lb_new), torch.abs(x_ub_new))
        xi_max_l2 = torch.linalg.norm(xi_max, ord=2, dim=1) # shape (N,)
        self.mesh_radius_l2 = torch.linalg.norm(x_ub_new[0] - x_lb_new[0], ord=2)/2.0

        N = len(x_lb_new)
        V_lb = torch.zeros(N, dtype=self.dtype)
        data_loader = DataLoader(x_m_new, batch_size=self.batch_size, shuffle=False)
        for i, x_m_batch in enumerate(data_loader):
            x_m_batch = x_m_batch.to(self.device)
            V_batch, V_dx_batch = self.V.forward_with_jacobian(x_m_batch) # shape (N, 1), (N, 1 ,n)
            V_batch = V_batch.squeeze(1) # shape (N,)
            V_dx_batch = V_dx_batch.squeeze(1) # shape (N, n)
            norm_V_dx_batch = torch.linalg.norm(V_dx_batch, ord=2, dim=1) # shape (N,)

            # Bound V value (minimization)
            V_lb_batch_zero_order = V_batch - self.lip_V_l2 * self.mesh_radius_l2 # shape (N,)
            V_lb_batch_first_order = V_batch - self.mesh_radius_l2 * norm_V_dx_batch - self.mesh_radius_l2**2 * self.hess_V_l2 / 2.0 # shape (N,)
            V_lb_batch = torch.max(V_lb_batch_zero_order, V_lb_batch_first_order) # shape (N,), keep the least conservative one

            V_lb[i*self.batch_size:min((i+1)*self.batch_size,N)] = V_lb_batch

        keep_idx = torch.where((xi_max_l2 > self.cutoff_radius) & (V_lb < self.V_c))[0]
        self.regions = TensorDataset(x_lb_new[keep_idx], 
                                     x_ub_new[keep_idx])
        return

