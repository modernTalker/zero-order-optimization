from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
from gradient_pruning import fast_random_mask_like
from .opt_utils import *
from collections import defaultdict

class ZO_clipped_SGD(ZeroOrderOptimizer):
    def __init__(
        self, params, 
        lr, eps,
        momentum=0, 
        clipping_type='norm', clipping_level=2.0,
        vector_sampling_type="standard_normal", perturbation_mode="two_side"
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            weight_decay=0.0,
            vector_sampling_type=vector_sampling_type,
            perturbation_mode=perturbation_mode,
        )

        type_to_default_level = {
            'no_clip': 0.0,
            'norm': 1.0,
            'layer_wise': 0.3,
            'coordinate_wise': 0.1
        }

        for group in self.param_groups:
            group['clipping_type'] = clipping_type
            group['clipping_level'] = clipping_level
            group['vector_sampling_type'] = vector_sampling_type

    def __setstate__(self, state):
        super(ZO_clipped_SGD, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is None:
            raise ValueError("Closure must be provided for zero-order optimization.")

        loss1, loss2 = None, None 
        self._prepare_parameters()
        
        self.zo_random_seed = np.random.randint(1_000_000_000)
        self.generator.manual_seed(self.zo_random_seed)
        
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = closure()
        self.generator.manual_seed(self.zo_random_seed)

        if self.perturbation_mode == "one_side":
            self.zo_perturb_parameters(scaling_factor=-1)
            self.generator.manual_seed(self.zo_random_seed)
            loss2 = closure()
            self.projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="one_side")
        elif self.perturbation_mode == "opf":
            self.zo_perturb_parameters(scaling_factor=-1)
            self.generator.manual_seed(self.zo_random_seed)
            loss2 = closure()
            self.projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="opf")
        elif self.perturbation_mode == "two_side":
            self.zo_perturb_parameters(scaling_factor=-2)
            loss2 = closure()
            self.projected_grad = self.grad_approx(loss_plus=loss1, loss_minus=loss2, perturbation_mode="two_side")
            self.generator.manual_seed(self.zo_random_seed)
            self.zo_perturb_parameters(scaling_factor=1)
            self.generator.manual_seed(self.zo_random_seed)
        else:
            raise ValueError("No perturbation mode provided.")
        
        self.generator.manual_seed(self.zo_random_seed)

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            lr = group['lr']
            momentum = group['momentum']
            clipping_type = group['clipping_type']
            clipping_level = group['clipping_level']

            for p in group['params']:
                device = p.device
                z = self.vector_sampler.sample(p.shape, generator=self.generator).to(device)
                d_p = (z * self.projected_grad) / group['eps']
                d_p_list.append(d_p)
                params_with_grad.append(p)
                
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

            # update parameters
            clipped_gradent_descent_step(
                params_with_grad, 
                d_p_list, 
                momentum_buffer_list,
                lr, 
                momentum,
                clipping_type, 
                clipping_level
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss1

        
def clipped_gradent_descent_step(
        params, 
        d_p_list, 
        momentum_buffer_list,
        lr: float,
        momentum: float,
        clipping_type: str,
        clipping_level: float):
    r"""Functional API that performs clipped step for slipped-SGD and clipped-SSTM algorithm 
        computation.
    See :class:`clipped_SGD` or class:`clipped_SSTM` for details.
    """
    grad_norm = 0.0
    if clipping_type == 'norm':
        for i in range(len(params)):
            grad_norm += (d_p_list[i].norm() ** 2).item()
        grad_norm = grad_norm ** 0.5
    
    for i, param in enumerate(params):
        d_p = d_p_list[i]

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1) # no dampening

            d_p = buf
                
        if clipping_type == 'no_clip':
            param.add_(d_p, alpha=-lr)
        elif clipping_type == 'norm':
            alpha = min(1.0, clipping_level / grad_norm)
            param.add_(d_p, alpha=-lr * alpha)
        elif clipping_type == 'layer_wise':
            layer_norm = d_p.norm().item()
            alpha = min(1.0, clipping_level / layer_norm)
            param.add_(d_p, alpha=-lr * alpha)
        elif clipping_type == 'coordinate_wise':
            eps = 1e-8
            alpha = torch.clip(clipping_level / (torch.abs(d_p) + eps), min=0, max=1)
            param.add_(-lr * alpha * d_p)