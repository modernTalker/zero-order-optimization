from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable

from .opt_utils import *

class Jaguar_SignSGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            beta: float = 0.9,
            lr: float = 0.01,
            eps: float = 1e-3,
            tensor_sampling_type: str = "standard_normal", 
            matrix_sampling_type: str = None, 
            perturbation_mode: str = "two_side"
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )
        
        for group in self.param_groups:
            group['beta'] = beta

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("ZO optimizers require a closure function.")

        loss1, loss2 = None, None 

        # 1. Standardized lazy state init (matches MUON)
        for group in self.param_groups:
            for param in group['params']:    
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_accum'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['step'] += 1

        # 2. Paired ZO perturbation
        self.zo_random_seed = int(np.random.randint(1_000_000_000))
        self.generator.manual_seed(self.zo_random_seed)

        self._indices_perturb(scaling_factor=1.0)
        loss1 = closure()
        
        self.generator.manual_seed(self.zo_random_seed)
        self._indices_perturb(scaling_factor=-2.0)
        loss2 = closure()
        
        self.generator.manual_seed(self.zo_random_seed)
        self._indices_perturb(scaling_factor=1.0)

        grad_update = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side")

        # 3. Update phase
        for group in self.param_groups:
            lr = group['lr']  
            beta = group['beta']
            eps = group['eps']
            grad_scalar = torch.tensor(grad_update / eps, device='cpu', dtype=torch.float32)
            
            for p in group['params']:
                state = self.state[p]
                indices = self._select_indices(param_shape=p.shape, device=p.device)
                
                if isinstance(indices, torch.Tensor):
                    state['grad_accum'][indices] = (
                        beta * state['grad_accum'][indices] + 
                        (1.0 - beta) * grad_scalar
                    )
                else:
                    rows, cols = indices
                    state['grad_accum'][rows[:, None], cols] = (
                        beta * state['grad_accum'][rows[:, None], cols] + 
                        (1.0 - beta) * grad_scalar
                    )
                
                update_direction = torch.sign(state['grad_accum'])
                p.data.add_(update_direction, alpha=-lr)
                
        return loss1