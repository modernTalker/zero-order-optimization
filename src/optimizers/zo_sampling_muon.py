from .base import ZeroOrderOptimizer
import torch
from torch.optim import SGD
import numpy as np
from .opt_utils import *

class ZO_SamplingMUON(ZeroOrderOptimizer):
    def __init__(self, params, args, gradient_sparsity=None):
        params = list(params)
        self._inner_optimizer = SGD(params, lr=args.learning_rate, momentum=args.momentum)
        super().__init__(params, args, gradient_sparsity)

    @torch.no_grad()
    def step(self, closure):
        args = self.args
        tau = args.zo_tau

        named_parameters_to_optim = []
        for name, param in self.named_parameters_all:
            if param.requires_grad:
                named_parameters_to_optim.append((name, param))
                param.grad = None  

        seed = np.random.randint(1_000_000_000)
        torch.manual_seed(seed)
        if hasattr(self, 'sparse_grad_rng'):
            self.sparse_grad_rng.manual_seed(seed)

        # Perturb
        def zo_muon_perturb_parameters(scaling_factor=1):
            for name, param in named_parameters_to_optim:
                # if name in E_dict:
                if param.ndim >= 2 and param.size(0) < 10000:
                    E = sample_ortho_approx(param.data.shape, device=self.device)
                    param.data.add_(tau * scaling_factor * E)
                else:
                    z = torch.randn_like(param)
                    mask = getattr(self, 'get_grad_sparsity_by_name', lambda x: None)(name)
                    if mask is not None:
                        z[fast_random_mask_like(z, mask, generator=self.sparse_grad_rng)] = 0
                    param.data.add_(tau * scaling_factor * z)

        # f(z_+)
        zo_muon_perturb_parameters(1)
        loss1 = closure()
        # f(z_-)
        zo_muon_perturb_parameters(-2)
        loss2 = closure()
        zo_muon_perturb_parameters(1)

        delta = (loss1 - loss2) / (2 * args.zo_eps)
        # rho = sign(f(z_+) - f(z_-))
        rho = torch.sign(loss1 - loss2)

        self._inner_optimizer.zero_grad()
        for name, param in named_parameters_to_optim:
            # if name in E_dict:
            if param.ndim >= 2 and param.size(0) < 10000:
                # _, U, V = E_dict[name]
                E = sample_ortho_approx(param.data.shape, device=self.device)
                grad = rho * E
            else:
                z = torch.randn_like(param)
                mask = getattr(self, 'get_grad_sparsity_by_name', lambda x: None)(name)
                if mask is not None:
                    z[fast_random_mask_like(z, mask, generator=self.sparse_grad_rng)] = 0
                grad = (rho if args.trainer=='zo_signsgd' else delta) * z
            param.grad = grad.to(param.dtype)

        self._inner_optimizer.step()

        return loss1