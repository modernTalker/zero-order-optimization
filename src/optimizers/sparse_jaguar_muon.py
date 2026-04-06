from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
import time

from .opt_utils import *

class Sparse_Jaguar_MUON(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            beta: float = 0.9,
            lr: float = 0.01,
            eps: float = 1e-3,
            tensor_sampling_type: str = "standard_normal", 
            matrix_sampling_type: str = None,  
            perturbation_mode: str = "two_side",
            params_ratio: float = 0.1
    ):
        super().__init__(
            params, lr=lr, eps=eps,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )

        self.params_ratio = params_ratio
        for group in self.param_groups:
            group['beta'] = beta

        self.all_params = [p for group in self.param_groups for p in group['params']]
        if len(self.all_params) == 0:
            raise ValueError("Optimizer received zero parameters.")

        for param in self.all_params:    
            state = self.state[param]
            if 'step' not in state:
                state['step'] = 0
                state['grad_accum'] = torch.zeros_like(param, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("ZO optimizers require a closure function to compute losses.")

        for group in self.param_groups:
            for param in group['params']:    
                self.state[param]['step'] += 1

        self.zo_random_seed = int(np.random.randint(1_000_000_000))
        self.generator.manual_seed(self.zo_random_seed)

        self._sparse_indices_perturb(scaling_factor=1.0, params_ratio=self.params_ratio)
        loss1 = closure()
        
        self.generator.manual_seed(self.zo_random_seed)
        self._sparse_indices_perturb(scaling_factor=-2.0, params_ratio=self.params_ratio)
        loss2 = closure()
        
        self.generator.manual_seed(self.zo_random_seed)
        self._sparse_indices_perturb(scaling_factor=1.0, params_ratio=self.params_ratio)

        grad_scalar = self.grad_approx(loss1, loss2, perturbation_mode="two_side")

        device = self.all_params[0].device
        n = max(1, int(len(self.all_params) * self.params_ratio))
        param_indices = torch.randperm(len(self.all_params), device=device, generator=self.generator)[:n]
        selected_param_ids = {id(self.all_params[idx]) for idx in param_indices}

        for group in self.param_groups:
            lr = group['lr']  
            beta = group['beta']
            eps = group['eps']
            
            for param in group['params']:     
                if id(param) not in selected_param_ids:
                    continue
                    
                state = self.state[param]                
                device = param.device
                
                s_type = self.state[param]["tensor_sampling_type"]
                z = self.tensor_sampler.sample(param.shape, generator=self.generator, sampler_type=s_type).to(device)
                grad_final = z * grad_scalar / eps 
                state['grad_accum'].mul_(beta).add_(grad_final, alpha=1.0 - beta)
                
                # MUON update: Newton-Schulz for matrices, SignSGD for vectors
                if param.ndim >= 2:
                    update_direction = zeropower_via_newtonschulz5(state['grad_accum'], steps=5)
                else:
                    update_direction = torch.sign(state['grad_accum'])
                    
                param.data.add_(update_direction, alpha=-lr)

        return loss1
    
    def _sparse_indices_perturb(self, scaling_factor=1.0, params_ratio=0.1):
        device = self.all_params[0].device
        n = max(1, int(len(self.all_params) * params_ratio))
        param_indices = torch.randperm(len(self.all_params), device=device, generator=self.generator)[:n]
        selected_param_ids = {id(self.all_params[idx]) for idx in param_indices}
        
        for group in self.param_groups:
            eps = group['eps']
            for param in group['params']:                     
                if id(param) not in selected_param_ids:
                    continue
                    
                s_type = self.state[param]["tensor_sampling_type"]
                z = self.tensor_sampler.sample(param.shape, generator=self.generator, sampler_type=s_type).to(param.device)
                param.data.add_(scaling_factor * eps * z)