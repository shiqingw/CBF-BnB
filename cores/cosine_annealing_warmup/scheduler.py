import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: 1.
        max_lr(float or list): First cycle's max learning rate. Default: 0.1.
        min_lr(float or list): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
        ):
        assert warmup_steps < first_cycle_steps, "warmup_steps must be less than first_cycle_steps"
        
        if not isinstance(max_lr, list):
            max_lr = [max_lr] * len(optimizer.param_groups)
        else:
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} max_lr, got {len(max_lr)}")
        
        if not isinstance(min_lr, list):
            min_lr = [min_lr] * len(optimizer.param_groups)
        else:
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} min_lr, got {len(min_lr)}")
        
        self.first_cycle_steps = first_cycle_steps  # First cycle step size
        self.cycle_mult = cycle_mult               # Cycle steps magnification
        self.base_max_lr = max_lr                  # List of initial max learning rates
        self.max_lr = max_lr.copy()                # Current max learning rates
        self.min_lr = min_lr                       # Min learning rates
        self.warmup_steps = warmup_steps           # Warmup step size
        self.gamma = gamma                         # Decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps   # Current cycle step size
        self.cycle = 0                             # Cycle count
        self.step_in_cycle = last_epoch            # Step in the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # Initialize learning rates
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for idx, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.min_lr[idx]
            self.base_lrs.append(self.min_lr[idx])
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            # Linear warm-up
            return [
                (max_lr_i - base_lr_i) * self.step_in_cycle / self.warmup_steps + base_lr_i
                for max_lr_i, base_lr_i in zip(self.max_lr, self.base_lrs)
            ]
        else:
            # Cosine annealing
            return [
                base_lr_i + (max_lr_i - base_lr_i) *
                (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) /
                (self.cur_cycle_steps - self.warmup_steps))) / 2
                for max_lr_i, base_lr_i in zip(self.max_lr, self.base_lrs)
            ]

    def step(self, epoch=None):
        if epoch is None:
            # Increment step in cycle
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                # Start new cycle
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult
                ) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                # Calculate cycle index
                if self.cycle_mult == 1.:
                    self.cycle = epoch // self.first_cycle_steps
                    self.step_in_cycle = epoch % self.first_cycle_steps
                else:
                    n = int(math.log(
                        (epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1),
                        self.cycle_mult
                    ))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) /
                        (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        # Update max_lr for the current cycle
        self.max_lr = [
            base_max_lr_i * (self.gamma ** self.cycle) for base_max_lr_i in self.base_max_lr
        ]
        self.last_epoch = math.floor(epoch)
        
        # Update learning rates for each parameter group
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
