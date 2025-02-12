import numpy as np
import torch
import platform

class Configuration(object):
    """Configuration class."""

    def __init__(self):
        """Initialization."""
        super(Configuration, self).__init__()

        # Dtype for computations
        self.np_dtype = np.float32
        self.pt_dtype = torch.float32
        self.dpi = 30
        self.pic_format = 'pdf'
        self.platform = platform.system().lower()
        
        if platform.system() == 'Darwin':
            # device = torch.device('cpu')
            # Comment out the following lines if you want to use MPS.
            # However, some functions are not supported by MPS.
            device = torch.device('cpu')
            # if not torch.backends.mps.is_available():
            #     device = torch.device('cpu')
            #     if not torch.backends.mps.is_built():
            #         print("MPS not available because the current PyTorch install was not "
            #         "built with MPS enabled.")
            #     else:
            #         print("MPS not available because the current MacOS version is not 12.3+ "
            #         "and/or you do not have an MPS-enabled device on this machine.")
            # else:
            #     device = torch.device('mps')
        else:
            # device = torch.device('cpu')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

