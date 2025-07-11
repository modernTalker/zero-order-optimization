from .base import ZeroOrderOptimizer
import torch
from torch.optim import SGD
import numpy as np
from .opt_utils import *

class ZO_Conserv(ZeroOrderOptimizer):
    def __init__(self, params, args, gradient_sparsity=None):
        params = list(params)
        self._inner_optimizer = SGD(params, lr=args.learning_rate, momentum=args.momentum)
        super().__init__(params, args, gradient_sparsity)

    @torch.no_grad()
    def step(self, closure):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        update in the conservative way, i.e. 
        reject the update if it's not decreasing
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in self.named_parameters_all:
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None  

        loss0 = closure()

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = closure()

        # Second function evaluation
        if self.args.perturbation_mode == "one_side":
            self.zo_perturb_parameters(scaling_factor=-1)
            loss2 = closure()
            self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
        else:  # two side perturbation
            self.zo_perturb_parameters(scaling_factor=-2)
            loss2 = closure()
            self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

            # Reset model back to its parameters at start of step
            self.zo_perturb_parameters(scaling_factor=1)

        def update_params(sign=1.0):
            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)

                if args.trainer == "zo_sign_opt":
                    # ----signOpt_orig
                    # TODo why do we multiply lr here? We will multiply lr twice?
                    graddiff_times_z = np.sign(self.projected_grad) * z
                    # ----signOpt_mul_sign
                    # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)
                else:
                    # ----mezo original
                    graddiff_times_z = self.projected_grad * z

                # # previous implementation
                # # no param.grad involved
                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()
                # more mem-efficient:
                # run optimizer.step here to avoid caching all grad.
                param.grad = sign * graddiff_times_z
                # self._inner_optimizer[name].step()
                self._inner_optimizer.step()
                # param.data = param.data - graddiff_times_z / args.q
                param.grad = None

        update_params()
        loss1 = closure()

        update_params(sign=-2.0)
        loss2 = closure()

        # conduct the update in the conservative way
        # choose from the three and take the minimum loss one
        if loss1 > loss0:
            if loss0 < loss2:
                update_params()
        else:
            if loss1 < loss2:
                update_params(2.0)

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1
