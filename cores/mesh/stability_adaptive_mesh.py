import torch
from torch.utils.data import TensorDataset, DataLoader

from .adaptive_mesh import AdaptiveMesh

class StabilityProbAdaptiveMesh(AdaptiveMesh):
    def __init__(self, f, lip_f_l2, V, lip_V_l2, x_lb, x_ub, batch_size, cutoff_radius, V_c, dtype, device):
        super().__init__(x_lb, x_ub, dtype)
        self.f = f
        self.lip_f_l2 = lip_f_l2
        self.V = V
        self.lip_V_l2 = lip_V_l2
        self.V_c = V_c
        self.batch_size = batch_size
        self.cutoff_radius = cutoff_radius
        self.device = device
    
    def bound(self):
        data_loader = DataLoader(self.regions, batch_size=self.batch_size, shuffle=False)
        f_lb = torch.zeros(len(self.regions), dtype=self.dtype)
        f_ub = torch.zeros(len(self.regions), dtype=self.dtype)
        for i, (x_lb_batch, x_ub_batch) in enumerate(data_loader):
            x_lb_batch = x_lb_batch.to(self.device)
            x_ub_batch = x_ub_batch.to(self.device)

            x_m_batch = (x_lb_batch + x_ub_batch) / 2.0
            x_l2_size_batch = torch.norm(x_ub_batch - x_lb_batch, p=2, dim=1) # shape (N,)
            f_lb_batch = self.f(x_m_batch).squeeze(-1) # shape (N,)
            f_ub_batch = f_lb_batch + self.lip_f_l2 * x_l2_size_batch/2.0 # shape (N,)

            V_ub_batch = self.V(x_m_batch).squeeze(-1) # shape (N,)
            V_lb_batch = V_ub_batch - self.lip_V_l2 * x_l2_size_batch/2.0 # shape (N,)

            # Set f_lb to -inf if x_m satisfies any of the two: 
            # 1. inside the cutoff radius and positive
            # 2. V_lb_batch > V_c and positive
            x_m_l2_batch = torch.norm(x_m_batch, p=2, dim=1) # shape (N,)
            f_lb_batch[torch.where((x_m_l2_batch <= self.cutoff_radius) & (f_lb_batch > 0))] = -float("inf")
            f_lb_batch[torch.where((V_lb_batch > self.V_c) & (f_lb_batch > 0))] = -float("inf")

            # Copy to f_lb and f_ub
            f_lb[i*self.batch_size:min((i+1)*self.batch_size,len(self.regions))] = f_lb_batch.detach().cpu()
            f_ub[i*self.batch_size:min((i+1)*self.batch_size,len(self.regions))] = f_ub_batch.detach().cpu()

        return f_lb, f_ub
    
    def refine(self, refine_idx):
        x_lb, x_ub = self.regions[refine_idx]
        x_lb_new, x_ub_new = self.split(x_lb, x_ub)
        x_m_new = (x_lb_new + x_ub_new) / 2.0
        xi_max = torch.max(torch.abs(x_lb_new), torch.abs(x_ub_new))
        xi_max_l2 = torch.norm(xi_max, p=2, dim=1) # shape (N,)
        x_l2_size = torch.norm(x_ub_new - x_lb_new, p=2, dim=1) # shape (N,)

        N = len(x_lb_new)
        V_lb = torch.zeros(N, dtype=self.dtype)
        data_loader = DataLoader(x_m_new, batch_size=self.batch_size, shuffle=False)
        for i, x_m_batch in enumerate(data_loader):
            x_m_batch = x_m_batch.to(self.device)
            V_ub_batch = self.V(x_m_batch).squeeze(-1).detach().cpu()
            V_lb_batch = V_ub_batch - self.lip_V_l2 * x_l2_size[i*self.batch_size:min((i+1)*self.batch_size,N)]/2.0
            V_lb[i*self.batch_size:min((i+1)*self.batch_size,N)] = V_lb_batch

        keep_idx = torch.where((xi_max_l2 > self.cutoff_radius) & (V_lb < self.V_c))[0]
        self.regions = TensorDataset(x_lb_new[keep_idx], 
                                     x_ub_new[keep_idx])
        return

