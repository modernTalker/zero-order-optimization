import torch
from torch.optim import Optimizer
from typing import Optional, Dict, Any, Union, Iterable, Callable
from .opt_utils import zeropower_via_newtonschulz5

class FO_MUON(Optimizer):
    def __init__(
        self, 
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
        lr: float = 0.01, 
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        super(FO_MUON, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
            
            if weight_decay != 0:
                for i, param in enumerate(params_with_grad):
                    d_p_list[i] = d_p_list[i].add(param, alpha=weight_decay)
            
            if momentum != 0:
                for i, param in enumerate(params_with_grad):
                    d_p = d_p_list[i]
                    if momentum_buffer_list[i] is not None:
                        buf = momentum_buffer_list[i]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf
                        momentum_buffer_list[i] = buf
                    else:
                        momentum_buffer_list[i] = d_p.clone().detach()
                        if nesterov:
                            d_p = d_p.add(momentum_buffer_list[i], alpha=momentum)
                        else:
                            d_p = momentum_buffer_list[i]
                    d_p_list[i] = d_p
            
            for i, param in enumerate(params_with_grad):
                grad = d_p_list[i]
                if param.ndim >= 2:
                    original_shape = grad.shape
                    if param.ndim > 2:
                        grad_2d = grad.view(-1, grad.shape[-1])
                    else:
                        grad_2d = grad
                    
                    grad_final_2d = zeropower_via_newtonschulz5(grad_2d, steps=5)
                    
                    if param.ndim > 2:
                        grad_final = grad_final_2d.view(original_shape)
                    else:
                        grad_final = grad_final_2d
                else:
                    grad_final = torch.sign(grad)
                d_p_list[i] = grad_final
            
            for i, param in enumerate(params_with_grad):
                param.add_(d_p_list[i], alpha=-lr)
                state = self.state[param]
                state['momentum_buffer'] = momentum_buffer_list[i]
        
        return loss
