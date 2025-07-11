from .base import ZeroOrderOptimizer
import torch
from torch.optim import SGD
import numpy as np
from .opt_utils import *

class Jaguar_SignSGD(ZeroOrderOptimizer):
    def __init__(self, params, args, gradient_sparsity=None):
        params = list(params)
        self._inner_optimizer = SGD(params, lr=args.learning_rate, momentum=args.momentum)
        super().__init__(params, args, gradient_sparsity)

    @torch.no_grad()
    def step(self, closure):
        args = self.args
        beta = args.zo_beta
        use_smoothing = args.zo_use_smoothing
        tau = args.zo_tau
        self.named_parameters_to_optim = []
        for name, param in self.named_parameters_all:
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None  
                
        self.zo_random_seed = np.random.randint(1_000_000_000)
        torch.manual_seed(self.zo_random_seed)

        selected_indices = {}
        original_values = {}

        self._inner_optimizer.zero_grad()
        for name, param in self.named_parameters_to_optim:
            if len(param.data.shape) == 1:
                n_elements = param.data.shape[0]
                k = max(1, n_elements // 10)
                indices = torch.randperm(n_elements, device=param.device)[:k]
                selected_indices[name] = indices
                original_values[name] = param.data[indices].clone()
            else:
                n_rows, n_cols = param.data.shape
                k = max(1, n_rows // 10)
                m = max(1, n_cols // 10)
                selected_rows = torch.randperm(n_rows, device=param.device)[:k]
                selected_cols = torch.randperm(n_cols, device=param.device)[:m]
                selected_indices[name] = (selected_rows, selected_cols)
                original_values[name] = param.data[selected_rows[:, None], selected_cols].clone()

        for name, param in self.named_parameters_to_optim:
            if len(param.data.shape) == 1:
                indices = selected_indices[name]
                param.data[indices] += tau
            else:
                selected_rows, selected_cols = selected_indices[name]
                param.data[selected_rows[:, None], selected_cols] += tau
        loss1 = closure()

        for name, param in self.named_parameters_to_optim:
            if len(param.data.shape) == 1:
                indices = selected_indices[name]
                param.data[indices] = original_values[name] - tau
            else:
                selected_rows, selected_cols = selected_indices[name]
                param.data[selected_rows[:, None], selected_cols] = original_values[name] - tau
        loss2 = closure()

        for name, param in self.named_parameters_to_optim:
            if len(param.data.shape) == 1:
                indices = selected_indices[name]
                param.data[indices] = original_values[name]
            else:
                selected_rows, selected_cols = selected_indices[name]
                param.data[selected_rows[:, None], selected_cols] = original_values[name]

        # rho = sign(f(z_+) - f(z_-))
        rho = (loss1 - loss2).item() / (2 * tau)

        for name, param in self.named_parameters_to_optim:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            else:
                param.grad.zero_()

            grad_update = rho
            if len(param.data.shape) == 1:
                indices = selected_indices[name]
                if use_smoothing:
                    param.grad[indices] = beta * param.grad[indices] + (1 - beta) * grad_update
                else:
                    param.grad[indices] = grad_update
            else:
                selected_rows, selected_cols = selected_indices[name]
                if use_smoothing:
                    param.grad[selected_rows[:, None], selected_cols] = beta * param.grad[selected_rows[:, None], selected_cols] + (1 - beta) * grad_update
                else:
                    param.grad[selected_rows[:, None], selected_cols] = grad_update

            param.grad = torch.sign(param.grad)

        self._inner_optimizer.step()

        assert args.gradient_accumulation_steps == 1
        return loss1
