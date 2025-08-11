from torch.optim import Optimizer
from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Dict, Any, Tuple, Union, Iterable
import torch
import numpy as np
from .opt_utils import VectorSampler
from gradient_pruning import fast_random_mask_like
from torch.optim import SGD
from collections import defaultdict

class ZeroOrderOptimizer(Optimizer, ABC):
    def __init__(self,
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            vector_sampling_type: str = "standard_normal",
            perturbation_mode: str = "two_side",
            device: str = "cuda", # FIXME: maybe change it
    ):
        """
        Base class for zero-order optimizers.

        Args:
            params: Model parameters to optimize:
                - Iterable[Tensor] (all parameters)
                - Iterable[Dict] (parameter gruops with different hyperparameters)
            lr: Learning rate, if None, then it has to be in parameter groups
            eps: Perturbation magnitude, if None, then it has to be in parameter groups
            momentum: Momentum factor, zero by default
            gradient_sparsity: Gradient sparsity (float for global or dict per parameter)
        """
        if lr is not None or eps is not None:
            defaults = {
                'lr': lr,
                'eps': eps,
                'momentum': momentum,
            }
        else:
            defaults = {'momentum': momentum}

        super().__init__(params, defaults)

        self._validate_hyperparameters()
        self.gradient_sparsity = gradient_sparsity

        self.state = defaultdict(dict)

        self.generator = torch.Generator(device=device)

        self.vector_sampler = VectorSampler(vector_sampling_type, device=device)
        self.perturbation_mode = perturbation_mode

        self.named_parameters_all = []
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group['params']):
                self.device = param.device
                param_name = f"group_{group_idx}.param_{param_idx}"
                self.named_parameters_all.append((param_name, param))
    
        self.zo_eps = self._calculate_zo_eps(eps=eps)

        self._inner_optimizers = None
        self._lr_schedulers = None

        # for group in self.param_groups:
        #     self._inner_optimizers.append(
        #         SGD(group['params'], lr=group['lr'], momentum=group['momentum'])
        #     )
       
    def set_lr_schedulers(self, lr_schedulers: List):
        """        
        Args:
            lr_schedulers: list of schedulers
        """
        self._lr_schedulers = lr_schedulers

    def _prepare_parameters(self) -> None:
        """Prepares parameters for optimization. Common for all optimizer's steps"""
        self.named_parameters_to_optim = [
            (name, param) for name, param in self.named_parameters_all 
            if param.requires_grad
        ]
        for _, param in self.named_parameters_to_optim:
            param.grad = None

    def _calculate_zo_eps(self, eps: Optional[float] = None):
        """"Estimates zo_eps for accurate grad approx as a weighted sum of all epsilons"""
        total_params = 0
        eps_sum = 0.0
        
        for group in self.param_groups:
            group_eps = group['eps']
            if group_eps is not None:
                group_params = sum(p.numel() for p in group['params'] if p.requires_grad)
                eps_sum += group_eps * group_params
                total_params += group_params
        
        return eps_sum / total_params if total_params > 0 else (eps if eps is not None else 1e-3)

    def _validate_hyperparameters(self):
        """Obligatory hyperparameters check"""
        required = ['lr', 'eps']
        for group in self.param_groups:
            for key in required:
                if key not in group:
                    raise ValueError(f"Missing required hyperparameter: {key}")
    
    @abstractmethod
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure: Callable that returns the loss and recomputes gradients.
        Returns:
            Loss tensor or None
        """
        pass
    
    def get_grad_sparsity_by_name(self, name: str) -> Optional[float]:
        """
        Get gradient sparsity for a parameter by name.

        Args:
            name: Parameter name
        Returns:
            Sparsity value or None
        """
        if self.gradient_sparsity is None:
            return None
        elif isinstance(self.gradient_sparsity, float):
            return self.gradient_sparsity
        elif isinstance(self.gradient_sparsity, dict):
            return self.gradient_sparsity[name]

    def zo_perturb_parameters(self, 
            random_seed: Optional[int] = None, 
            scaling_factor: float = 1.0
    ) -> None:
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """
        self.zo_random_seed = random_seed if random_seed is not None else np.random.randint(1000000000)
        self.generator.manual_seed(self.zo_random_seed)

        sparsity_dict = {}
        for name, param in self.named_parameters_all:
            if param.requires_grad:
                sparsity_dict[id(param)] = self.get_grad_sparsity_by_name(name)

        self.perturb_parameters(
            scaling_factor=scaling_factor,
            random_seed=self.zo_random_seed,
            sparsity_dict=sparsity_dict,
            element_wise=True
        )
    
    def perturb_parameters(
        self, 
        scaling_factor: float = 1.0,
        random_seed: Optional[int] = None,
        custom_perturb_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        indices: Optional[Dict[int, Tuple[str, Any]]] = None,
        element_wise: bool = False,
        sparsity_dict: Optional[Dict[int, float]] = None
    ) -> None:
        """
        Applies perturbation to parameters, either globally or to selected indices.
        
        Args:
            scaling_factor: Scale of perturbation
            random_seed: Fixes random seed for reproducibility
            generator: Custom random number generator
            custom_perturb_func: Custom perturbation function
            indices: Dictionary of indices from _select_perturbation_indices for selective perturbation
            element_wise: Whether to apply perturbations element-wise (for indices mode)
            sparsity_dict: {param_id: sparsity} for gradient sparsity
        """
        if random_seed is None:
            random_seed = np.random.randint(1000000)
        
        self.generator.manual_seed(random_seed)
        
        if sparsity_dict is not None:
            original_perturb_func = custom_perturb_func
            def sparse_perturb_func(param: torch.Tensor) -> torch.Tensor:
                if original_perturb_func:
                    z = original_perturb_func(param)
                else:
                    z = self.vector_sampler.sample(param.shape, generator=self.generator)
                
                param_id = id(param)
                self.generator.manual_seed(random_seed)
                if param_id in sparsity_dict:
                    sparsity = sparsity_dict[param_id]
                    if sparsity is not None:
                        mask = fast_random_mask_like(z, sparsity, generator=self.generator)
                        self.generator.manual_seed(random_seed)
                        z[mask] = 0
                return z
            custom_perturb_func = sparse_perturb_func
            
        for group in self.param_groups:
            eps = group['eps']
            for p in group['params']:
                if not p.requires_grad:
                    continue

                param_id = id(p)
                perturb = None
                if custom_perturb_func:
                    pass 

                elif indices is not None and param_id in indices:
                    spec = indices[param_id]
                    
                    if spec[0] == '1d':
                        idx = spec[1]
                        if element_wise:
                            perturb = self.vector_sampler.sample(p.data[idx].shape, generator=self.generator) * eps
                        else:
                            perturb = torch.ones_like(p.data[idx]) * eps

                        p.data[idx].add_(scaling_factor * perturb)

                    elif spec[0] == '2d':
                        rows, cols = spec[1], spec[2]
                        if element_wise:
                            slice_data = p.data[rows[:, None], cols]
                            perturb = self.vector_sampler.sample(slice_data.shape, generator=self.generator) * eps
                        else:
                            perturb = torch.ones_like(p.data[rows[:, None], cols]) * eps
                        p.data[rows[:, None], cols].add_(scaling_factor * perturb)
                
                else:
                    if perturb is None:
                        z = self.vector_sampler.sample(p.shape, generator=self.generator)
                        perturb = z * eps
                    p.data.add_(scaling_factor * perturb)

    def grad_approx(
        self,
        loss_original: torch.Tensor,
        loss_perturbed: torch.Tensor,
        perturbation_mode: str = "two_side"
    ) -> float:
        """
        Aproximates gradient.
        
        Args:
            loss_original: Loss function value in a source point
            loss_perturbed: Loss function value is a perturbated point
            perturbation_mode: 'one_side' or 'two_side'
            
        Returns:
            Gradient estimation
        """
        if perturbation_mode == "one_side":
            return ((loss_perturbed - loss_original) / self.zo_eps).item()
        elif perturbation_mode == "two_side":
            return ((loss_perturbed - loss_original) / (2 * self.zo_eps)).item()
        else:
            raise ValueError(f"Unknown perturbation mode: {perturbation_mode}")
    
    def _get_flat_params(self) -> List[torch.Tensor]:
        """Returns full list of parameters copy"""
        return [p.detach().clone() for group in self.param_groups for p in group['params']]
    
    def _set_flat_params(self, params: List[torch.Tensor]) -> None:
        """Setes parameters from List"""
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                p.data.copy_(params[idx])
                idx += 1
                
    def _select_perturbation_indices(
        self,
        row_frac: float = 0.1,
        col_frac: float = 0.1,
        min_elements: int = 1
    ) -> Dict[int, Tuple[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Selects random perturbation indices
        
        Args:
            row_frac: Fraction of rows for perturbation (for 2D+ tensors)
            col_frac: Fraction of columns for perturbation (for 2D+ tensors)
            min_elements: Minimum number of elements for perturbtion
            
        Returns:
            Dictionary with indices for each parameter
        """
        indices = {}
        for group in self.param_groups:
            for p in group['params']:
                # if p.requires_grad:
                param_id = id(p)
                
                if p.dim() == 1:
                    n = p.size(0)
                    k = max(min_elements, int(n * row_frac))
                    idx = torch.randperm(n)[:k]
                    indices[param_id] = ('1d', idx)
                    
                elif p.dim() >= 2:
                    n, m = p.size(0), p.size(1)
                    k = max(min_elements, int(n * row_frac))
                    l = max(min_elements, int(m * col_frac))
                    
                    rows = torch.randperm(n)[:k]
                    cols = torch.randperm(m)[:l]
                    indices[param_id] = ('2d', rows, cols)
                        
        return indices
    
    def _apply_sparse_perturbation(self, indices_dict, scaling_factor):
        for group in self.param_groups:
            for param in group['params']:
                param_id = id(param)
                if param_id not in indices_dict:
                    continue
                    
                indices_info = indices_dict[param_id]
                perturbation = self.zo_eps * scaling_factor
                
                if indices_info[0] == '1d':
                    indices = indices_info[1]
                    param.data[indices] += perturbation
                    
                elif indices_info[0] == '2d':
                    rows, cols = indices_info[1], indices_info[2]
                    param.data[rows[:, None], cols] += perturbation