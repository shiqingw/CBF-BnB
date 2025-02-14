import torch
from torch.utils.data import TensorDataset, DataLoader

from .adaptive_mesh import AdaptiveMesh

class InclusionAdaptiveMeshSecondOrder(AdaptiveMesh):
    def __init__(self, f, lip_f_l2, hess_f_l2, thrid_order_elementwise_l2, x_lb, x_ub, batch_size, dtype, device):
        super().__init__(x_lb, x_ub, dtype)
        self.f = f
        self.lip_f_l2 = lip_f_l2
        self.hess_f_l2 = hess_f_l2
        self.thrid_order_elementwise_l2 = thrid_order_elementwise_l2
        self.batch_size = batch_size
        self.device = device
        self.mesh_radius_l1 = torch.linalg.norm(x_ub - x_lb, ord=1)/2.0
        self.mesh_radius_l2 = torch.linalg.norm(x_ub - x_lb, ord=2)/2.0

    def bound(self):
        data_loader = DataLoader(self.regions, batch_size=self.batch_size, shuffle=False)
        f_lb = torch.zeros(len(self.regions), dtype=self.dtype)
        f_ub = torch.zeros(len(self.regions), dtype=self.dtype)

        for i, (x_lb_batch, x_ub_batch) in enumerate(data_loader):
            x_lb_batch = x_lb_batch.to(self.device)
            x_ub_batch = x_ub_batch.to(self.device)
            x_m_batch = (x_lb_batch + x_ub_batch) / 2.0
            c_vec_batch = (x_ub_batch - x_lb_batch) / 2.0

            # Value, gradient, and Hessian of V
            V_batch, V_dx_batch, V_dxdx_batch = self.f.forward_with_jacobian_and_hessian_method2(x_m_batch) # shape (N,1), (N, 1, n), (N, 1, n, n)
            V_batch = V_batch.squeeze(1) # shape (N,)
            V_dx_batch = V_dx_batch.squeeze(1) # shape (N, n)
            V_dxdx_batch = V_dxdx_batch.squeeze(1) # shape (N, n, n)
            abs_V_dx_batch = torch.abs(V_dx_batch) # shape (N, n)
            norm_l2_V_dxdx_batch = torch.linalg.norm(V_dxdx_batch, ord=2, dim=(1,2)) # shape (N,)
            abs_V_dx_c_vec_batch = torch.sum(abs_V_dx_batch * c_vec_batch, dim=1) # shape (N,)

            # Zero-order bound
            f_lb_zero_order_batch = V_batch.clone()
            f_ub_zero_order_batch = V_batch + self.lip_f_l2 * self.mesh_radius_l2

            # First-order bound
            ## Lower bound
            f_lb_first_order_batch = V_batch + abs_V_dx_c_vec_batch - self.mesh_radius_l2**2 * self.hess_f_l2 / 2.0
            ## Upper bound
            f_ub_first_order_batch = V_batch + abs_V_dx_c_vec_batch + self.mesh_radius_l2**2 * self.hess_f_l2 / 2.0

            # Second-order bound
            ## Lower bound
            f_lb_second_order_batch = V_batch.clone()
            f_lb_second_order_batch += abs_V_dx_c_vec_batch - self.mesh_radius_l2**2 * self.thrid_order_elementwise_l2 * self.mesh_radius_l1 / 6.0
            sign_V_dx_batch = torch.sign(V_dx_batch) # shape (N, n)
            sign_V_dx_c_vec_batch = sign_V_dx_batch * c_vec_batch # shape (N, n)
            f_lb_second_order_batch += 0.5 * torch.bmm(torch.bmm(sign_V_dx_c_vec_batch.unsqueeze(1), V_dxdx_batch), sign_V_dx_c_vec_batch.unsqueeze(2)).squeeze(2).squeeze(1)
            
            ## Upper bound
            f_ub_second_order_batch = V_batch + abs_V_dx_c_vec_batch + self.mesh_radius_l2**2 * norm_l2_V_dxdx_batch / 2.0 \
                                        + self.mesh_radius_l2**2 * self.thrid_order_elementwise_l2 * self.mesh_radius_l1 / 6.0

            # Choose the best bound
            f_ub_batch = torch.min(torch.min(f_ub_zero_order_batch, f_ub_first_order_batch), f_ub_second_order_batch)
            f_lb_batch = torch.max(torch.max(f_lb_zero_order_batch, f_lb_first_order_batch), f_lb_second_order_batch)

            # Copy to f_lb and f_ub
            f_lb[i*self.batch_size:min((i+1)*self.batch_size,len(self.regions))] = f_lb_batch.detach().cpu()
            f_ub[i*self.batch_size:min((i+1)*self.batch_size,len(self.regions))] = f_ub_batch.detach().cpu()

        return f_lb, f_ub

    def refine(self, refine_idx):
        x_lb, x_ub = self.regions[refine_idx]
        x_lb_new, x_ub_new = self.split(x_lb, x_ub)
        self.regions = TensorDataset(x_lb_new, 
                                     x_ub_new)
        self.mesh_radius_l1 = torch.linalg.norm(self.regions[0][0] - self.regions[0][1], ord=1)/2.0
        self.mesh_radius_l2 = torch.linalg.norm(self.regions[0][0] - self.regions[0][1], ord=2)/2.0
        return
