import torch
from torch.utils.data import TensorDataset, DataLoader

class AdaptiveMesh:
    def __init__(self, x_lb, x_ub, dtype):
        assert len(x_lb) == len(x_ub)
        if torch.any(x_lb > x_ub):
            raise ValueError(f"Lower bound {x_lb} should be less than upper bound {x_ub}")
        self.regions = TensorDataset(x_lb.cpu(),
                                     x_ub.cpu())
        self.dtype = dtype 

    def split(self, x_lb, x_ub):
        N, D = x_lb.shape
        x_delta = x_ub - x_lb  # shape (N, D)
        _, max_indices = torch.max(x_delta, dim=1)  # shape (N,)

        indices = torch.arange(N)
        mid_points = (x_lb[indices, max_indices] + x_ub[indices, max_indices]) / 2.0  # shape (N,)

        x_lb_new = torch.zeros(2 * N, D, dtype=self.dtype)
        x_ub_new = torch.zeros(2 * N, D, dtype=self.dtype)
        
        # First half: left regions
        x_lb_new[0:N] = x_lb
        x_ub_new[0:N] = x_ub
        x_ub_new[indices, max_indices] = mid_points  # Update upper bounds of the left regions

        # Second half: right regions
        x_lb_new[N:2*N] = x_lb
        x_ub_new[N:2*N] = x_ub
        x_lb_new[N + indices, max_indices] = mid_points  # Update lower bounds of the right regions

        return x_lb_new, x_ub_new
    
    def refine(self, refine_idx):
        x_lb, x_ub = self.regions[refine_idx]
        x_lb_new, x_ub_new = self.split(x_lb, x_ub)
        self.regions = TensorDataset(x_lb_new, x_ub_new)

        return True
