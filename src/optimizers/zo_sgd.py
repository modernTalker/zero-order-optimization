from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
from gradient_pruning import fast_random_mask_like
from .opt_utils import *
from collections import defaultdict

class ZO_SGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            vector_sampling_type: str = "standard_normal",
            perturbation_mode: str = "two_side",
            q: int = 1,
            module_wise_perturbation: bool = False,
            coordinate_perturbation: bool = False
    ):
        defaults = dict(
            lr=lr,
            eps=eps,
            momentum=momentum,
            vector_sampling_type=vector_sampling_type
        )
        super().__init__(params, defaults)
        
        self.state = defaultdict(dict)
        self.perturbation_mode = perturbation_mode
        self.lr = lr 
        self.module_wise_perturbation = module_wise_perturbation
        self.coordinate_perturbation = coordinate_perturbation
        self.projected_grad = None
        self.zo_random_seed = None

    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None 
        self._prepare_parameters()
        random_seeds = []
        
        seed = np.random.randint(1_000_000_000)
        random_seeds.append(seed)
        self.generator.manual_seed(seed)
        
        self.zo_perturb_parameters(scaling_factor=1, random_seed=seed)
        loss1 = closure()
        self.generator.manual_seed(seed)

        if self.perturbation_mode == "one_side":
            self.zo_perturb_parameters(scaling_factor=-1, random_seed=seed)
            self.generator.manual_seed(seed)
            loss2 = closure()
            self.projected_grad = (loss2 - loss1).item()
        else:
            self.zo_perturb_parameters(scaling_factor=-2, random_seed=seed)
            loss2 = closure()
            self.projected_grad = (loss2 - loss1).item() / 2
            self.generator.manual_seed(seed)
            self.zo_perturb_parameters(scaling_factor=1, random_seed=seed)
            self.generator.manual_seed(seed)
            
        self._apply_gradients(random_seeds=random_seeds)
        return loss1 
    
    @torch.no_grad()
    def _apply_gradients(self, random_seeds = None) -> None:
        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                if not any(name for name, p in self.named_parameters_to_optim if p is param):
                    continue
                
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                
                device = param.device
                grad = torch.zeros_like(param)
                eps = 1 # FIXME: do we need it?
                
                for seed in random_seeds:
                    self.generator.manual_seed(seed)
                    z = self.vector_sampler.sample(param.shape, generator=self.generator).to(device)
                    name = next(name for name, p in self.named_parameters_to_optim if p is param)
                    sparsity = self.get_grad_sparsity_by_name(name)
                    if sparsity is not None:
                        mask = fast_random_mask_like(z, sparsity, generator=self.sparse_grad_rng).to(device)
                        z[mask] = 0
                    
                    grad += (z * self.projected_grad * eps) / len(random_seeds)
                
                param.data.add_(grad, alpha=-self.lr)

    def _get_module_parameters(self):
        return [("all", [(name, p) for name, p in self.named_parameters_all if p.requires_grad])]
