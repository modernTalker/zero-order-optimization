from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
from gradient_pruning import fast_random_mask_like
from .opt_utils import *
from collections import defaultdict

class ZO_clipped_SSTM(ZeroOrderOptimizer):
    def __init__(
        self, params, 
        lr, L, eps,
        clipping_type='norm', clipping_level=2.0, 
        nu=1, a_k_ratio_upper_bound=1.0, clipping_iter_start=None,
        vector_sampling_type="standard_normal", perturbation_mode="two_side"
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            momentum=0.0,  # No momentum in clipped_SSTM
            weight_decay=0.0,  # No weight decay in clipped_SSTM
            vector_sampling_type=vector_sampling_type,
            perturbation_mode=perturbation_mode,
        )


        type_to_default_level = {
            'no_clip': 0.0,
            'norm': 1.0,
            'layer_wise': 0.3,
            'coordinate_wise': 0.1
        }
        clipping_level = type_to_default_level[clipping_type]
        if nu < 0.0 or nu > 1.0:
            raise ValueError("Invalid nu: {}".format(nu))
        if a_k_ratio_upper_bound <= 0.0 or a_k_ratio_upper_bound > 1.0:
            raise ValueError("Invalid a_k_ratio_upper_bound: {}".format(a_k_ratio_upper_bound))
        if clipping_iter_start is not None:
            if not isinstance(clipping_iter_start, int) or clipping_iter_start <= 0:
                raise ValueError("Invalid clipping_iter_start: {}, should be positive integer")
            if (nu > 0 and clipping_type == 'norm'):
                a = 1 / lr
                clipping_level = 1 / (2 * a * L) * (clipping_iter_start + 1) ** (2 * nu / (1 + nu))
            elif (nu < 1e-4):
                a = 1 / lr
                clipping_level = clipping_level / (2 * a * L)

        for group in self.param_groups:
            group['L'] = L
            group['clipping_type'] = clipping_type
            group['clipping_level'] = clipping_level
            group['nu'] = nu
            group['a_k_ratio_upper_bound'] = a_k_ratio_upper_bound
            group['vector_sampling_type'] = vector_sampling_type
            group['state'] = dict()

    def __setstate__(self, state):
        super(ZO_clipped_SSTM, self).__setstate__(state)

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
            a = 1 / group['lr']
            L = group['L']
            clipping_type = group['clipping_type']
            clipping_level = group['clipping_level']
            nu = group["nu"]
            a_k_ratio_upper_bound = group["a_k_ratio_upper_bound"]

            state = group['state']
            # lazy state initialization
            if len(state) == 0:
                state['k'] = 0
                state['alpha_k_1'] = 0
                state['lambda_k_1'] = 0
                state['A_k'] = 0
                state['A_k_1'] = 0

                state['y_k'] = []
                state['z_k'] = []
                for p in group['params']:
                    state['y_k'].append(p.detach().clone())
                    state['z_k'].append(p.detach().clone())
                
            k = state['k']
            alpha_k_1 = state['alpha_k_1']
            lambda_k_1 = state['lambda_k_1']
            A_k = state['A_k']
            A_k_1 = state['A_k_1']
            y_k = [y.detach().clone() for y in state['y_k']]
            z_k = [z.detach().clone() for z in state['z_k']]

            if k > 0:
                d_p_list = []
                for p in group['params']:
                    device = p.device
                    z = self.vector_sampler.sample(p.shape, generator=self.generator).to(device)
                    d_p = (z * self.projected_grad) / group['eps']
                    d_p_list.append(d_p)

                # update z_{k+1}
                clipped_gradent_descent_step(
                    z_k, 
                    d_p_list, 
                    None, # no momentum history
                    alpha_k_1, 
                    0, # no momentum, thus 0
                    clipping_type, 
                    lambda_k_1
                )

                # update y_{k+1}
                i = 0
                for p in group['params']:
                    y_k[i].data = (A_k * y_k[i].data + alpha_k_1 * z_k[i].data) / A_k_1
                    i += 1

            # k_1 means "k + 1", so alpha_k_1 means \alpha_{k+1}
            alpha_k_1 = 1 / (2 * a * L) * (k + 1) ** (2 * nu / (1 + nu))

            A_k = state['A_k_1']
            A_k_1 = A_k + alpha_k_1

            # apply upper bound on A_k / A_{k+1} ratio
            if a_k_ratio_upper_bound < 1.0:
                ratio_mul_factor = 1.0 / (1.0 - a_k_ratio_upper_bound)
                if A_k > ratio_mul_factor * alpha_k_1:
                    A_k = (ratio_mul_factor - 1.0) * alpha_k_1
                    A_k_1 = ratio_mul_factor * alpha_k_1

            lambda_k_1 = clipping_level / alpha_k_1
            # lambda_k_1 = clipping_level
            
            state['y_k'] = y_k
            state['z_k'] = z_k

            # update x_{k+1}
            i = 0
            for p in group['params']:
                p.data = (A_k * state['y_k'][i].data + alpha_k_1 * state['z_k'][i].data) / A_k_1
                i += 1

            state['k'] += 1
            state['alpha_k_1'] = alpha_k_1
            state['lambda_k_1'] = lambda_k_1
            state['A_k'] = A_k
            state['A_k_1'] = A_k_1

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
            alpha = min(1, clipping_level / grad_norm)
            param.add_(d_p, alpha=-lr*alpha)
        elif clipping_type == 'layer_wise':
            alpha = min(1, clipping_level / d_p.norm())
            param.add_(d_p, alpha=-lr*alpha)
        elif clipping_type == 'coordinate_wise':
            eps = 1e-8
            alpha = torch.clip(clipping_level / (torch.abs(d_p) + eps), min=0, max=1)
            param.add_(-lr * alpha * d_p)
