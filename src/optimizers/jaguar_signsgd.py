from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
import time

from .opt_utils import *

class Jaguar_SignSGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            # tau: float = 0.01,
            beta: float = 0.9,
            use_smoothing: bool = True,
            lr: float = 0.01,
            eps: float = 1e-3,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            perturbation_mode: str = "two_side",
            q: int = 1,
            module_wise_perturbation: bool = False,
            coordinate_perturbation: bool = False
    ):
        defaults = dict(
            lr=lr,
            eps=eps,
            momentum=momentum,
            # tau=tau,
            beta=beta,
            use_smoothing=use_smoothing,
            gradient_sparsity=gradient_sparsity
        )
        super().__init__(params, defaults)
        
        # self.tau = tau
        self.lr = lr 
        self.beta = beta
        self.use_smoothing = use_smoothing
        self.perturbation_mode = perturbation_mode
        self.q = q
        self.module_wise_perturbation = module_wise_perturbation
        self.coordinate_perturbation = coordinate_perturbation

    @torch.no_grad()
    def step(self, closure):
        # self._prepare_parameters()
        print(f"START STEP: {self.state['step']}")
        st_time = time.time()
        for group in self.param_groups:
            for param in group['params']:
                # if param.grad is None:
                    # continue
                    
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_accum'] = torch.zeros_like(
                        param, 
                        memory_format=torch.preserve_format
                    )
                state['step'] += 1
        end_time = time.time()
        print("State cycle time: {}".format(end_time - st_time))

        self.zo_random_seed = np.random.randint(1_000_000_000)
        self.generator.manual_seed(self.zo_random_seed)

        st_time = time.time()
        selected_indices = self._select_perturbation_indices(row_frac=0.1, col_frac=0.1, min_elements=1)
        self.generator.manual_seed(self.zo_random_seed)
        end_time = time.time()
        print("Select indices time: {}".format(end_time - st_time))


        st_time = time.time()
        self._apply_sparse_perturbation(selected_indices, scaling_factor=1)
        loss1 = closure()
        end_time = time.time()
        print("First perturb time: {}".format(end_time - st_time))


        st_time = time.time()
        self._apply_sparse_perturbation(selected_indices, scaling_factor=-2)
        loss2 = closure()
        end_time = time.time()
        print("Second perturb time: {}".format(end_time - st_time))

        st_time = time.time()
        self._apply_sparse_perturbation(selected_indices, scaling_factor=1)
        end_time = time.time()
        print("Third perturb time: {}".format(end_time - st_time))
        # grad_update = (loss1 - loss2).item() / (2 * self.zo_eps)

        grad_update = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, perturbation_mode=self.perturbation_mode)

        print("PARAM GROUPS ", len(self.param_groups))

        st_time = time.time()
        for group in self.param_groups:
            for param in group['params']:
                # name = next(name for name, p in self.named_parameters_to_optim if p is param)
                state = self.state[param]
                indices = selected_indices[id(param)]
                
                if self.use_smoothing:
                    # if isinstance(indices, torch.Tensor):
                    if indices[0] == '1d':
                        idx = indices[1]
                        state['grad_accum'][idx] = (
                            self.beta * state['grad_accum'][idx] + 
                            (1 - self.beta) * grad_update
                        )
                    else:
                        rows, cols = indices[1], indices[2]
                        state['grad_accum'][rows[:, None], cols] = (
                            self.beta * state['grad_accum'][rows[:, None], cols] + 
                            (1 - self.beta) * grad_update
                        )
                else:
                    # if isinstance(indices, torch.Tensor):
                    if indices[0] == '1d':
                        state['grad_accum'][indices] = grad_update
                    else:
                        rows, cols = indices[1], indices[2]
                        state['grad_accum'][rows[:, None], cols] = grad_update
                
                update_direction = torch.sign(state['grad_accum'])
                # param.data.add_(update_direction, alpha=-group['lr']
                param.data.add_(update_direction, alpha=-self.lr)
        end_time = time.time()
        print("Last update time: {}".format(end_time - st_time))

        return loss1
