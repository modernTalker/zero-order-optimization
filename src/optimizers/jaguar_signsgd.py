from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
import time

from .opt_utils import *

class Jaguar_SignSGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            beta: float = 0.9,
            use_smoothing: bool = True,
            lr: float = 0.01,
            eps: float = 1e-3,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            perturbation_mode: str = "two_side",
            q: int = 1,
            module_wise_perturbation: bool = False,
            coordinate_perturbation: bool = False,
            vector_sampling_type: str = "standard_normal", 
            matrix_sampling_type: str = None,  
            device: str = "cuda" 
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            gradient_sparsity=gradient_sparsity,
            vector_sampling_type=vector_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
            device=device
        )
        
        for group in self.param_groups:
            group['beta'] = beta
            group['use_smoothing'] = use_smoothing
        
        self.q = q
        self.module_wise_perturbation = module_wise_perturbation
        self.coordinate_perturbation = coordinate_perturbation

    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None 
        self._prepare_parameters()  

        for group in self.param_groups:
            for param in group['params']:    
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_accum'] = torch.zeros_like(
                        param, 
                        memory_format=torch.preserve_format
                    )
                state['step'] += 1

        self.zo_random_seed = np.random.randint(1_000_000_000)
        self.generator.manual_seed(self.zo_random_seed)

        self._indices_perturb(scaling_factor = 1.0)
        if closure is not None:
            loss1 = closure()
        self.generator.manual_seed(self.zo_random_seed)

        self._indices_perturb(scaling_factor = -2.0)
        if closure is not None:
            loss2 = closure()
        self.generator.manual_seed(self.zo_random_seed)

        self._indices_perturb(scaling_factor = 1.0)
        self.generator.manual_seed(self.zo_random_seed)

        grad_update = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side")

        for group in self.param_groups:
            lr = group['lr']  
            beta = group['beta']
            use_smoothing = group['use_smoothing']

            for param in group['params']:
                if not any(name for name, p in self.named_parameters_to_optim if p is param):
                    continue
                state = self.state[param]
                indices = self._select_indices(param_shape=param.shape, device=param.device)
                
                if use_smoothing:
                    if isinstance(indices, torch.Tensor):
                        state['grad_accum'][indices] = (
                            beta * state['grad_accum'][indices] + 
                            (1 - beta) * grad_update
                        )
                    else:
                        rows, cols = indices
                        state['grad_accum'][rows[:, None], cols] = (
                            beta * state['grad_accum'][rows[:, None], cols] + 
                            (1 - beta) * grad_update
                        )
                else:
                    if isinstance(indices, torch.Tensor):
                        state['grad_accum'][indices] = grad_update
                    else:
                        rows, cols = indices
                        state['grad_accum'][rows[:, None], cols] = grad_update
                
                update_direction = torch.sign(state['grad_accum'])
                param.data.add_(update_direction, alpha=-lr)
                
        return loss1
