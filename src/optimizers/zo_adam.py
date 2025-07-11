from .base import ZeroOrderOptimizer
import torch
from torch.optim import Adam
import numpy as np
from .opt_utils import *

class ZO_Adam(ZeroOrderOptimizer):
    def __init__(self, trainer, params, defaults):
        params = list(params)
        
        self._inner_optimizer = Adam(params, lr=defaults["lr"])
        super().__init__(trainer, params, defaults)

    @torch.no_grad()
    def step(self, model, inputs):
        args = self.trainer.args
        if args.module_wise_perturbation:
            assert args.q == 1, "module-wise perturbation only supports q=1"
            if args.coordinate_perturbation:
                return self.zo_step_with_module_wise_perturbation_coordinate(model, inputs)
            else:
                return self.zo_step_with_module_wise_perturbation(model, inputs)
        elif args.q == 1:
            return self.zo_step(model, inputs)
        elif args.q > 1:
            return self.zo_step_v1(model, inputs)
        else:
            raise ValueError(f"q={args.q} is not supported.")

    @torch.no_grad()
    def zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.trainer.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None  # Make sure the grad is empty and will not be updated.

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.trainer.zo_forward(model, inputs)

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for _ in range(args.q):  # TODO: shall we change the seed?
            if self.trainer.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.trainer.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.trainer.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.trainer.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.trainer.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)
                grad_sparsity = self.trainer.get_grad_sparsity_by_name(name)
                if grad_sparsity is not None:
                    z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

                if args.trainer == "zo_sign_opt":
                    # ----signOpt_orig
                    # TODO: why do we multiply lr here? We will multiply lr twice?
                    graddiff_times_z = np.sign(self.projected_grad) * z
                else:
                    # ----mezo original
                    graddiff_times_z = self.projected_grad * z

                # # previous implementation
                # # no param.grad involved
                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # more mem-efficient:
                # run optimizer.step here to avoid caching all grad.
                param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                self._inner_optimizer.step()  # will only update grad that is not None.
                param.grad = None  # avoid further update.
                
        # No gradient accumulation support
        assert self.trainer.args.gradient_accumulation_steps == 1

        return loss1
    
    @torch.no_grad()
    def zo_step_with_module_wise_perturbation_coordinate(self, model, inputs):
        """Update the parameters right after perturbing the parameters."""
        args = self.trainer.args
        perturbed_module_level = args.perturbed_module_level

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        all_losses = []

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for module_name, module in self.grouped_module_iter(model, perturbed_module_level):
            self.named_parameters_to_optim = []
            for name, param in module.named_parameters():
                if param.requires_grad:
                    self.named_parameters_to_optim.append((f"{module_name}.{name}", param))
                    param.grad = None  # Make sure the grad is empty and will not be updated.

            self.zo_perturb_parameters(scaling_factor=1)
            loss1 = self.trainer.zo_forward(model, inputs)

            all_losses.append(loss1)

            for _ in range(args.q):
                if self.trainer.args.perturbation_mode == "one_side":
                    self.zo_perturb_parameters(scaling_factor=-1)
                    loss2 = self.trainer.zo_forward(model, inputs)
                    self.projected_grad = ((loss1 - loss2) / self.trainer.args.zo_eps).item()
                else:  # two side perturbation
                    self.zo_perturb_parameters(scaling_factor=-2)
                    loss2 = self.trainer.zo_forward(model, inputs)
                    self.projected_grad = ((loss1 - loss2) / (2 * self.trainer.args.zo_eps)).item()

                    # Reset model back to its parameters at start of step
                    self.zo_perturb_parameters(scaling_factor=1)

                # Set the random seed to ensure that we sample the same z for perturbation/update
                torch.manual_seed(self.zo_random_seed)
                self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
                for name, param in self.named_parameters_to_optim:
                    # Resample z
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                     dtype=param.data.dtype)
                    grad_sparsity = self.trainer.get_grad_sparsity_by_name(name)
                    if grad_sparsity is not None:
                        z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

                    if args.trainer == "zo_sign_opt":
                        graddiff_times_z = np.sign(self.projected_grad) * z
                    else:
                        graddiff_times_z = self.projected_grad * z

                    param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                    self.optimizer.step()  # will only update grad that is not None.
                    param.grad = None  # avoid further update.

        assert self.trainer.args.gradient_accumulation_steps == 1

        print(f"[debugging] num blocks: {len(all_losses)}")

        return torch.stack(all_losses).mean()

    @torch.no_grad()
    def zo_step_with_module_wise_perturbation(self, model, inputs):
        """Update all parameters once after perturbing all the parameters."""
        args = self.trainer.args
        perturbed_module_level = args.perturbed_module_level

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        all_losses = []
        module_name_to_projected_grads = {}

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for module_name, module in self.grouped_module_iter(model, perturbed_module_level):
            self.named_parameters_to_optim = []
            for name, param in module.named_parameters():
                if param.requires_grad:
                    self.named_parameters_to_optim.append((f"{module_name}.{name}", param))
                    param.grad = None  # Make sure the grad is empty and will not be updated.

            self.zo_perturb_parameters(scaling_factor=1)
            loss1 = self.trainer.zo_forward(model, inputs)

            all_losses.append(loss1)

            if self.trainer.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.trainer.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.trainer.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.trainer.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.trainer.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            module_name_to_projected_grads[module_name] = self.projected_grad

        for module_name, module in self.grouped_module_iter(model, perturbed_module_level):
            self.named_parameters_to_optim = []
            for name, param in module.named_parameters():
                if param.requires_grad:
                    self.named_parameters_to_optim.append((f"{module_name}.{name}", param))
                    param.grad = None  # Make sure the grad is empty and will not be updated.

            self.projected_grad = module_name_to_projected_grads[module_name]

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)
                grad_sparsity = self.trainer.get_grad_sparsity_by_name(name)
                if grad_sparsity is not None:
                    z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

                if args.trainer == "zo_sign_opt":
                    graddiff_times_z = np.sign(self.projected_grad) * z
                else:
                    graddiff_times_z = self.projected_grad * z

                param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                self.optimizer.step()  # will only update grad that is not None.
                param.grad = None  # avoid further update.

        assert self.trainer.args.gradient_accumulation_steps == 1

        print(f"[debugging] num blocks: {len(all_losses)}")

        return torch.stack(all_losses).mean()

    
    @torch.no_grad()
    def zo_step_v1(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        Works with q > 1. But for q > 1, it is not memory efficient.
        """
        args = self.trainer.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO: avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)

        for i_q in range(args.q):  # TODO: shall we change the seed?
            # Sample the random seed for sampling z
            self.zo_random_seed = np.random.randint(1000000000)

            # First function evaluation
            self.zo_perturb_parameters(scaling_factor=1)
            loss1 = self.trainer.zo_forward(model, inputs)

            # Second function evaluation
            if self.trainer.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.trainer.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.trainer.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.trainer.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.trainer.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)

                if args.trainer == "zo_sign_opt":
                    # ----signOpt_orig
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
                if i_q == 0:
                    param.grad = graddiff_times_z / args.q
                else:
                    param.grad += graddiff_times_z / args.q
                # if i_q == args.q - 1:
                #     self.optimizer.step()  # TODO: If q > 1, We cannot use this trick anymore. This will cause repeated update.
                #     # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                #     param.grad = None

        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q
        self.optimizer.step()
        self.optimizer.zero_grad()

        # No gradient accumulation support
        assert self.trainer.args.gradient_accumulation_steps == 1

        return loss1

    @torch.no_grad()
    def zo_step_v2(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        Works with q > 1. But for q > 1, it is not memory efficient.
        """
        args = self.trainer.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO: avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)

        seed_list = []
        projected_grad_list = []
        for i_q in range(args.q):  # TODO: shall we change the seed?
            # Sample the random seed for sampling z
            self.zo_random_seed = np.random.randint(1000000000)
            seed_list.append(self.zo_random_seed)

            # First function evaluation
            self.zo_perturb_parameters(scaling_factor=1)
            loss1 = self.trainer.zo_forward(model, inputs)

            # Second function evaluation
            if self.trainer.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.trainer.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.trainer.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.trainer.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.trainer.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            projected_grad_list.append(self.projected_grad)

        # difference from v1: switch the order of for loop
        # to save memory
        for name, param in self.named_parameters_to_optim:
            for i_q in range(args.q):
                # Set the random seed to ensure that we sample the same z for perturbation/update
                torch.manual_seed(seed_list[i_q])

                graddiff_times_z = torch.zeros_like(param.data, device=param.data.device,
                                                    dtype=param.data.dtype)

                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)

                if args.trainer == "zo_sign_opt":
                    # ----signOpt_orig
                    graddiff_times_z += np.sign(projected_grad_list[i_q]) * z
                    # ----signOpt_mul_sign
                    # graddiff_times_z = torch.sign(projected_grad_list[i_q] * z)
                else:
                    # ----mezo original
                    graddiff_times_z += projected_grad_list[i_q] * z

                # # previous implementation
                # # no param.grad involved
                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()
                # more mem-efficient:
                # run optimizer.step here to avoid caching all grad.
                if i_q == args.q - 1:
                    param.grad = graddiff_times_z.detach()
                    self.optimizer[name].step()
                    # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                    param.grad = None

        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q

        # No gradient accumulation support
        assert self.trainer.args.gradient_accumulation_steps == 1

        return loss1
