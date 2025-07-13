from .base import ZeroOrderOptimizer
import torch
from torch.optim import SGD
import numpy as np
from .opt_utils import *
from typing import Optional, Callable, Dict, Any, Union, List, Iterable, Tuple
from gradient_pruning import fast_random_mask_like

class ZO_SGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            perturbation_mode: str = "two_side",
            q: int = 1,
            module_wise_perturbation: bool = False,
            coordinate_perturbation: bool = False
    ):
        """
        Zero-Order Stochastic Gradient Descent optimizer (MeZO implementation).
        
        Args:
            params: Parameters to optimize (can specify per-group hyperparameters)
            lr: Optional base learning rate (must be specified in groups if None)
            eps: Optional base perturbation (must be specified in groups if None)
            momentum: Momentum factor (default: 0.0)
            gradient_sparsity: Gradient sparsity (float or dict per parameter)
            perturbation_mode: 'one_side' or 'two_side' (default: 'two_side')
            q: Number of random directions (default: 1)
            module_wise_perturbation: Whether to perturb modules separately
            coordinate_perturbation: Whether to update immediately after perturbation
        """
        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            gradient_sparsity=gradient_sparsity
        )
        # params = list(params)
        self.perturbation_mode = perturbation_mode
        self.q = q
        self.module_wise_perturbation = module_wise_perturbation
        self.coordinate_perturbation = coordinate_perturbation

        # inner optimizer for each param_gropus
        self._inner_optimizers = []
        for group in self.param_groups:
            self._inner_optimizers.append(
                SGD([group['params']], lr=group['lr'], momentum=group['momentum'])
            )
        # TODO: Maybe we need a random seed in the input
        self.projected_grad: Optional[float] = None
        self.zo_random_seed: Optional[int] = None

    @torch.no_grad()
    def step(self, closure):
        """ 
        Performs a single optimization step.

        Args:
            closure: Callable that returns the loss and recomputes gradients.
        Returns:
            Loss tensor or None
        """
        # NOTE: In the original code, the zo_step_v2 isn't used anywhere
        args = self.args
        if args.module_wise_perturbation:
            assert args.q == 1, "module-wise perturbation only supports q=1"
            if args.coordinate_perturbation:
                return self.zo_step_with_module_wise_perturbation_coordinate(closure)
            return self.zo_step_with_module_wise_perturbation(closure)
        elif args.q == 1:
            return self.zo_step(closure)
        elif args.q > 1:
            return self.zo_step_v1(closure)
        else:
            raise ValueError(f"q={args.q} is not supported.")

    @torch.no_grad()
    def zo_step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args
        self._prepare_parameters()
        
        # What parameters to optimize
        # NOTE: This part is in the function self._prepare_parameters()
        # self.named_parameters_to_optim = []
        # for name, param in self.named_parameters_all:
            # if param.requires_grad:
                # self.named_parameters_to_optim.append((name, param))
                # param.grad = None  

        # Sample the random seed for sampling z
        # FIXME: We definitly shoul fix the seed :)
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = closure()

        # Second function evaluation
        # NOTE: Since q == 1 for the method we need the loop no more
        assert args.q == 1, "Only support q=1 for the memory efficiency."
        # for _ in range(args.q):  # TODO: shall we change the seed?
        if self.args.perturbation_mode == "one_side":
            self.zo_perturb_parameters(scaling_factor=-1)
            loss2 = closure()
            self.projected_grad = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, perturbation_mode="one_side")
        else:  # two side perturbation
            self.zo_perturb_parameters(scaling_factor=-2)
            loss2 = closure()
            # FIXME: May be we can calculate weighted_avg(eps) to estimate the gradient correctly
            # self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
            self.projected_grad = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, perturbation_mode="two_side")

            # Reset model back to its parameters at start of step
            self.zo_perturb_parameters(scaling_factor=1)
        
        # NOTE: The while functional is common for all steps and is now in self._apply_gradients()
        self._apply_gradients()

        # Set the random seed to ensure that we sample the same z for perturbation/update
        # torch.manual_seed(self.zo_random_seed)
        # self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
        # for name, param in self.named_parameters_to_optim:
        #     # Resample z
        #     z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
        #                         dtype=param.data.dtype)
        #     grad_sparsity = self.get_grad_sparsity_by_name(name)
        #     if grad_sparsity is not None:
        #         z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

        #     if args.trainer == "zo_sign_opt":
        #         # ----signOpt_orig
        #         # TODO: why do we multiply lr here? We will multiply lr twice?
        #         graddiff_times_z = np.sign(self.projected_grad) * z
        #     else:
        #         # ----mezo original
        #         graddiff_times_z = self.projected_grad * z

        #     # # previous implementation
        #     # # no param.grad involved
        #     # param.data -= self._get_learning_rate() * self.projected_grad * z

        #     # more mem-efficient:
        #     # run optimizer.step here to avoid caching all grad.
        #     param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
        #     self._inner_optimizer.step()  # will only update grad that is not None.
        #     param.grad = None  # avoid further update.
                
        # # No gradient accumulation support
        # assert self.args.gradient_accumulation_steps == 1

        return loss1
    
    @torch.no_grad()
    def zo_step_with_module_wise_perturbation_coordinate(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """
        Update the parameters right after perturbing the parameters.
        Module-wise perturbation with immediate updates.
        """
        # args = self.args
        # perturbed_module_level = args.perturbed_module_level

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # all_losses = []

        losses = []

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.

        for _, module_params in self._get_module_parameters():
            self.named_parameters_to_optim = module_params
            loss = self._module_perturbation_step(closure)
            losses.append(loss)
            self._apply_gradients()  # therefore the changes are here
        
        # Second function evaluation
        # assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        # for module_name, module in self.grouped_module_iter(model, perturbed_module_level):
        #     self.named_parameters_to_optim = []
        #     for name, param in self.named_parameters_all:
        #         if param.requires_grad:
        #             self.named_parameters_to_optim.append((name, param))
        #             param.grad = None  

        #     self.zo_perturb_parameters(scaling_factor=1)
        #     loss1 = closure()

        #     all_losses.append(loss1)

        #     for _ in range(args.q):
        #         if self.args.perturbation_mode == "one_side":
        #             self.zo_perturb_parameters(scaling_factor=-1)
        #             loss2 = closure()
        #             self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
        #         else:  # two side perturbation
        #             self.zo_perturb_parameters(scaling_factor=-2)
        #             loss2 = closure()
        #             self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

        #             # Reset model back to its parameters at start of step
        #             self.zo_perturb_parameters(scaling_factor=1)

        #         # Set the random seed to ensure that we sample the same z for perturbation/update
        #         torch.manual_seed(self.zo_random_seed)
        #         self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
        #         for name, param in self.named_parameters_to_optim:
        #             # Resample z
        #             z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
        #                              dtype=param.data.dtype)
        #             grad_sparsity = self.get_grad_sparsity_by_name(name)
        #             if grad_sparsity is not None:
        #                 z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

        #             if args.trainer == "zo_sign_opt":
        #                 graddiff_times_z = np.sign(self.projected_grad) * z
        #             else:
        #                 graddiff_times_z = self.projected_grad * z

        #             param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
        #             self.optimizer.step()  # will only update grad that is not None.
        #             param.grad = None  # avoid further update.

        # assert self.args.gradient_accumulation_steps == 1

        # print(f"[debugging] num blocks: {len(all_losses)}")

        return torch.stack(losses).mean()

    @torch.no_grad()
    def zo_step_with_module_wise_perturbation(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Update all parameters once after perturbing all the parameters."""
        # args = self.args
        # perturbed_module_level = args.perturbed_module_level
    
        losses = []
        grad_dict = {}

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # all_losses = []
        # module_name_to_projected_grads = {}

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.

        for module_name, module_params in self._get_module_parameters():
            self.named_parameters_to_optim = module_params
            loss = self._module_perturbation_step(closure)
            losses.append(loss)
            grad_dict[module_name] = self.projected_grad
        
        for module_name, module_params in self._get_module_parameters():
            self.named_parameters_to_optim = module_params
            self.projected_grad = grad_dict[module_name]
            self._apply_gradients()
        
        # TODO: Got rid of huge code part, needs carefull checks
        # Second function evaluation
        # assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        # for module_name, module in self.grouped_module_iter(model, perturbed_module_level):
        #     self.named_parameters_to_optim = []
        #     for name, param in self.named_parameters_all:
        #         if param.requires_grad:
        #             self.named_parameters_to_optim.append((name, param))
        #             param.grad = None  
                    
        #     self.zo_perturb_parameters(scaling_factor=1)
        #     loss1 = closure()

        #     all_losses.append(loss1)

        #     if self.args.perturbation_mode == "one_side":
        #         self.zo_perturb_parameters(scaling_factor=-1)
        #         loss2 = closure()
        #         self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
        #     else:  # two side perturbation
        #         self.zo_perturb_parameters(scaling_factor=-2)
        #         loss2 = closure()
        #         self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

        #         # Reset model back to its parameters at start of step
        #         self.zo_perturb_parameters(scaling_factor=1)

        #     module_name_to_projected_grads[module_name] = self.projected_grad

        # for module_name, module in self.grouped_module_iter(model, perturbed_module_level):
        #     self.named_parameters_to_optim = []
        #     for name, param in module.named_parameters():
        #         if param.requires_grad:
        #             self.named_parameters_to_optim.append((f"{module_name}.{name}", param))
        #             param.grad = None  # Make sure the grad is empty and will not be updated.

        #     self.projected_grad = module_name_to_projected_grads[module_name]

        #     # Set the random seed to ensure that we sample the same z for perturbation/update
        #     torch.manual_seed(self.zo_random_seed)
        #     self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
        #     for name, param in self.named_parameters_to_optim:
        #         # Resample z
        #         z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
        #                          dtype=param.data.dtype)
        #         grad_sparsity = self.get_grad_sparsity_by_name(name)
        #         if grad_sparsity is not None:
        #             z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

        #         if args.trainer == "zo_sign_opt":
        #             graddiff_times_z = np.sign(self.projected_grad) * z
        #         else:
        #             graddiff_times_z = self.projected_grad * z

        #         param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
        #         self.optimizer.step()  # will only update grad that is not None.
        #         param.grad = None  # avoid further update.

        # assert self.args.gradient_accumulation_steps == 1

        # print(f"[debugging] num blocks: {len(all_losses)}")

        return torch.stack(losses).mean()

    
    @torch.no_grad()
    def zo_step_v1(self, closure):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        Multi-direction gradient estimation (q > 1).
        """
        # args = self.args

        self._prepare_parameters()
        projected_grads = []

        # What parameters to optimize
        # NOTE: This part is in the function self._prepare_parameters()
        # self.named_parameters_to_optim = []
        # for name, param in self.named_parameters_all:
            # if param.requires_grad:
                # self.named_parameters_to_optim.append((name, param))
                # param.grad = None  

        for i_q in range(self.q):  # TODO: shall we change the seed?
            # Sample the random seed for sampling z
            # self.zo_random_seed = np.random.randint(1000000000)

            seed = np.random.randint(1000000000)

            # First function evaluation
            self.zo_perturb_parameters(scaling_factor=1, random_seed=seed)
            loss1 = closure()

            # Second function evaluation
            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = closure()
                grad = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, perturbation_mode="one_side")
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2, random_seed=seed)
                loss2 = closure()
                grad = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, perturbation_mode="two_side")

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1, random_seed=seed)

            projected_grads.append(grad)

            # we alwaya return loss1, now need to return loss1 from the first itersstion
            if i_q == 0:
                first_loss = loss1
            
        self.projected_grad = sum(projected_grads) / self.q
        self._apply_gradients(random_seeds=[np.random.randint(1000000000) for _ in range(self.q)])

            # Set the random seed to ensure that we sample the same z for perturbation/update
        #     torch.manual_seed(self.zo_random_seed)
        #     for name, param in self.named_parameters_to_optim:
        #         # Resample z
        #         z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
        #                          dtype=param.data.dtype)

        #         if args.trainer == "zo_sign_opt":
        #             # ----signOpt_orig
        #             graddiff_times_z = np.sign(self.projected_grad) * z
        #             # ----signOpt_mul_sign
        #             # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)
        #         else:
        #             # ----mezo original
        #             graddiff_times_z = self.projected_grad * z

        #         # # previous implementation
        #         # # no param.grad involved
        #         # param.data -= self._get_learning_rate() * self.projected_grad * z

        #         # param.grad += graddiff_times_z.detach()
        #         # more mem-efficient:
        #         # run optimizer.step here to avoid caching all grad.
        #         if i_q == 0:
        #             param.grad = graddiff_times_z / args.q
        #         else:
        #             param.grad += graddiff_times_z / args.q
        #         # if i_q == args.q - 1:
        #         #     self.optimizer.step()  # TODO: If q > 1, We cannot use this trick anymore. This will cause repeated update.
        #         #     # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
        #         #     param.grad = None

        # # for name, param in self.named_parameters_to_optim:
        # #     param.grad = param.grad / args.q
        # self.optimizer.step()
        # self.optimizer.zero_grad()

        # # No gradient accumulation support
        # assert self.args.gradient_accumulation_steps == 1

        return first_loss

    # @torch.no_grad()
    # def zo_step_v2(self, closure):
    #     """
    #     Estimate gradient by MeZO. Return the loss from f(theta + z)
    #     Works with q > 1. But for q > 1, it is not memory efficient.
    #     """
    #     args = self.args

    #     # What parameters to optimize
    #     self.named_parameters_to_optim = []
    #     for name, param in self.named_parameters_all:
    #         if param.requires_grad:
    #             self.named_parameters_to_optim.append((name, param))
    #             param.grad = None  

    #     seed_list = []
    #     projected_grad_list = []
    #     for i_q in range(args.q):  # TODO: shall we change the seed?
    #         # Sample the random seed for sampling z
    #         self.zo_random_seed = np.random.randint(1000000000)
    #         seed_list.append(self.zo_random_seed)

    #         # First function evaluation
    #         self.zo_perturb_parameters(scaling_factor=1)
    #         loss1 = closure()

    #         # Second function evaluation
    #         if self.args.perturbation_mode == "one_side":
    #             self.zo_perturb_parameters(scaling_factor=-1)
    #             loss2 = closure()
    #             self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
    #         else:  # two side perturbation
    #             self.zo_perturb_parameters(scaling_factor=-2)
    #             loss2 = closure()
    #             self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

    #             # Reset model back to its parameters at start of step
    #             self.zo_perturb_parameters(scaling_factor=1)

    #         projected_grad_list.append(self.projected_grad)

    #     # difference from v1: switch the order of for loop
    #     # to save memory
    #     for name, param in self.named_parameters_to_optim:
    #         for i_q in range(args.q):
    #             # Set the random seed to ensure that we sample the same z for perturbation/update
    #             torch.manual_seed(seed_list[i_q])

    #             graddiff_times_z = torch.zeros_like(param.data, device=param.data.device,
    #                                                 dtype=param.data.dtype)

    #             # Resample z
    #             z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
    #                                 dtype=param.data.dtype)

    #             if args.trainer == "zo_sign_opt":
    #                 # ----signOpt_orig
    #                 graddiff_times_z += np.sign(projected_grad_list[i_q]) * z
    #                 # ----signOpt_mul_sign
    #                 # graddiff_times_z = torch.sign(projected_grad_list[i_q] * z)
    #             else:
    #                 # ----mezo original
    #                 graddiff_times_z += projected_grad_list[i_q] * z

    #             # # previous implementation
    #             # # no param.grad involved
    #             # param.data -= self._get_learning_rate() * self.projected_grad * z

    #             # param.grad += graddiff_times_z.detach()
    #             # more mem-efficient:
    #             # run optimizer.step here to avoid caching all grad.
    #             if i_q == args.q - 1:
    #                 param.grad = graddiff_times_z.detach()
    #                 self.optimizer[name].step()
    #                 # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
    #                 param.grad = None

    #     # for name, param in self.named_parameters_to_optim:
    #     #     param.grad = param.grad / args.q

    #     # No gradient accumulation support
    #     assert self.args.gradient_accumulation_steps == 1

    #     return loss1
    
    def _prepare_parameters(self) -> None:
        """Prepares parameters for optimization. Common for all optimizer's steps"""
        self.named_parameters_to_optim = [
            (name, param) for name, param in self.named_parameters_all 
            if param.requires_grad
        ]
        for _, param in self.named_parameters_to_optim:
            param.grad = None
    
    def _apply_gradients(self, random_seeds: Optional[List[int]] = None) -> None:
        """
        Applies gradients using per-group hyperparameters.
        
        Args:
            random_seeds: List of seeds for perturbation vectors (q > 1)
        """
        if random_seeds is None:
            random_seeds = [self.zo_random_seed]
            
        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                if not any(name for name, p in self.named_parameters_to_optim if p is param):
                    continue
                
                grad = torch.zeros_like(param)
                eps = group['eps']
                
                for seed in random_seeds:
                    torch.manual_seed(seed)
                    self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
                    
                    z = torch.normal(mean=0, std=1, size=param.shape, device=param.device)
                    name = next(name for name, p in self.named_parameters_to_optim if p is param)
                    sparsity = self.get_grad_sparsity_by_name(name)
                    if sparsity is not None:
                        z[fast_random_mask_like(z, sparsity, generator=self.sparse_grad_rng)] = 0
                    
                    grad += (self.projected_grad * z * eps) / len(random_seeds)
                
                param.grad = grad
                self._inner_optimizers[group_idx].step()
                param.grad = None
    
    def _module_perturbation_step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Performs single module perturbation step."""
        self.zo_random_seed = np.random.randint(1000000000)
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = closure()
        
        if self.perturbation_mode == "one_side":
            self.zo_perturb_parameters(scaling_factor=-1)
            loss2 = closure()
            self.projected_grad = (loss1 - loss2).item()
        else:  # two_side
            self.zo_perturb_parameters(scaling_factor=-2)
            loss2 = closure()
            self.projected_grad = ((loss1 - loss2) / 2).item()
            self.zo_perturb_parameters(scaling_factor=1)
        
        return loss1

    def _get_module_parameters(self) -> List[Tuple[str, List[Tuple[str, torch.Tensor]]]]:
        """
        Groups parameters by module (simplified implementation).
        
        Returns:
            List of (module_name, parameters) tuples
        """
        # NOTE: maybe need to find another replacement for modules
        return [("all", [(name, p) for name, p in self.named_parameters_all if p.requires_grad])]
