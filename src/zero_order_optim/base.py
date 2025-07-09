from torch.optim import Optimizer
from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Dict, Any, Tuple, Union
import torch

class ZeroOrderOptimizer(Optimizer, ABC):
    def __init__(self, params, defaults: Dict[str, Any]):
        super().__init__(params, defaults)
        self._validate_hyperparameters()
        
    def _validate_hyperparameters(self):
        """Obligatory hyperparameters check"""
        required = ['lr', 'eps']
        for group in self.param_groups:
            for key in required:
                if key not in group:
                    raise ValueError(f"Missing required hyperparameter: {key}")
    
    @abstractmethod
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Proceeds one optimization step"""
        pass
    
    def perturb_parameters(
        self, 
        scaling_factor: float = 1.0,
        random_seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        custom_perturb_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        indices: Optional[Dict[int, Tuple[str, Any]]] = None,
        element_wise: bool = False
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
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
            
        for group in self.param_groups:
            eps = group['eps']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                    
                param_id = id(p)
                
                if indices is not None and param_id in indices:
                    spec = indices[param_id]
                    
                    if spec[0] == '1d':
                        idx = spec[1]
                        if element_wise:
                            if custom_perturb_func:
                                perturbations = custom_perturb_func(p.data[idx]) * eps
                            else:
                                perturbations = torch.randn_like(p.data[idx], generator=generator) * eps
                            p.data[idx] += scaling_factor * perturbations
                        else:
                            p.data[idx] += scaling_factor * eps
                            
                    elif spec[0] == '2d':
                        rows, cols = spec[1], spec[2]
                        if element_wise:
                            slice_data = p.data[rows[:, None], cols]
                            if custom_perturb_func:
                                perturbations = custom_perturb_func(slice_data) * eps
                            else:
                                perturbations = torch.randn_like(slice_data, generator=generator) * eps
                            p.data[rows[:, None], cols] += scaling_factor * perturbations
                        else:
                            p.data[rows[:, None], cols] += scaling_factor * eps
                
                else:
                    if custom_perturb_func:
                        perturbation = custom_perturb_func(p) * eps
                    else:
                        z = torch.randn_like(p, generator=generator)
                        perturbation = z * eps
                        
                    p.data.add_(scaling_factor * perturbation)
    
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
            return (loss_perturbed - loss_original).item()
        elif perturbation_mode == "two_side":
            return ((loss_perturbed - loss_original) / 2).item()
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
                if p.requires_grad:
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