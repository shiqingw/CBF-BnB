import torch
import torch.nn as nn
import numpy as np

class SingleIntegrator3D(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32):
        super(SingleIntegrator3D, self).__init__()

        self.state_dim = 3
        self.control_dim = 3
        self.dtype = dtype

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:

        return u