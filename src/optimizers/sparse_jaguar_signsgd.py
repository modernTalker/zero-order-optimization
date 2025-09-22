from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
import time

from .opt_utils import *

class Sparse_Jaguar_SignSGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            beta: float = 0.9,
            lr: float = 0.01,
            eps: float = 1e-3,
            vector_sampling_type: str = "standard_normal", 
            matrix_sampling_type: str = None,  
            perturbation_mode: str = "two_side",
            params_ratio: float = 0.1,
            rows_ratio: float = 1.0,
            columns_ratio: float = 1.0
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            vector_sampling_type=vector_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )

        self.params_ratio = params_ratio
        self.rows_ratio = rows_ratio
        self.columns_ratio = columns_ratio
        
        for group in self.param_groups:
            group['beta'] = beta

        self.all_params = [p for group in self.param_groups for p in group['params']]

    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None 
        # self._prepare_parameters()  

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

        self._sparse_indices_perturb(scaling_factor = 1.0, params_ratio = self.params_ratio, rows_ratio=self.rows_ratio, cols_ratio=self.columns_ratio)
        if closure is not None:
            loss1 = closure()
        self.generator.manual_seed(self.zo_random_seed)

        self._sparse_indices_perturb(scaling_factor = -2.0, params_ratio = self.params_ratio, rows_ratio=self.rows_ratio, cols_ratio=self.columns_ratio)
        if closure is not None:
            loss2 = closure()
        self.generator.manual_seed(self.zo_random_seed)

        self._sparse_indices_perturb(scaling_factor = 1.0, params_ratio = self.params_ratio, rows_ratio=self.rows_ratio, cols_ratio=self.columns_ratio)
        self.generator.manual_seed(self.zo_random_seed)

        grad_update = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side")

        n = max(1, int(len(self.all_params) * self.params_ratio))
        param_indices =  torch.randperm(len(self.all_params), device=self.device, generator=self.generator)[:n]
        self.generator.manual_seed(self.zo_random_seed)
        selected_param_ids = {id(self.all_params[idx]) for idx in param_indices}
        for group in self.param_groups:
            lr = group['lr']  
            beta = group['beta']
            eps = group['eps']
            
            for param in group['params']:  
                if id(param) not in selected_param_ids:
                    continue
                if not param.requires_grad:
                    print("ALARM")
                    continue
                state = self.state[param]
                indices = self._select_indices(param_shape=param.shape, cols_ratio=self.columns_ratio, rows_ratio=self.rows_ratio, device=param.device)
                
                device = param.device
                z = self.vector_sampler.sample(param.shape, generator=self.generator).to(device)
                grad_final = z * grad_update / eps 
                
                if isinstance(indices, torch.Tensor):
                    state['grad_accum'] = (
                        beta * state['grad_accum'] + 
                        (1 - beta) * grad_final
                    )
                else:
                    # rows, cols = indices
                    state['grad_accum'] = (
                        beta * state['grad_accum'] + 
                        (1 - beta) * grad_final
                    )
                
                update_direction = torch.sign(state['grad_accum'])
                param.data.add_(update_direction, alpha=-lr)

                # if id(param) == id(self.all_params[param_indices[0]]):
                #     print(1)
                #     print(param.data)
                #     print(2)
                #     print(grad_final)
                #     print(3)
                #     print(state['grad_accum'])
                #     print(4)
                #     print(update_direction)
                #     print("--------------")
                
        return loss1
    
    def _sparse_indices_perturb(self, scaling_factor = 1.0, params_ratio = 0.1, rows_ratio = 1.0, cols_ratio = 1.0):
        n = max(1, int(len(self.all_params) * params_ratio))
        param_indices =  torch.randperm(len(self.all_params), device=self.device, generator=self.generator)[:n]
        self.generator.manual_seed(self.zo_random_seed)
        for idx in param_indices:
            param = self.all_params[idx]
            # indices = self._select_indices(param_shape=param.shape, rows_ratio=rows_ratio, cols_ratio=cols_ratio, device=param.device)
            # if isinstance(indices, torch.Tensor):
            #     param.data += scaling_factor * self.zo_eps
            # else:
            #     rows, cols = indices
            device = param.device
            z = self.vector_sampler.sample(param.shape, generator=self.generator).to(device)
            param.data += self.zo_eps * z
