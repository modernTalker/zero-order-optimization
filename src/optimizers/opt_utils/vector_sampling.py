import torch

class VectorSampler:
    def __init__(self, sampler_type, device):
        self.sampler_type = sampler_type
        self.device = device
    
    def sample(self, param_shape):
        if self.sampler_type == "standard_normal":
            self.sample = self._standard_normal
        else:
            raise NotImplementedError(f"Sampling {self.sampler_type} is not implemented")

        return self.sample(param_shape)

    def _standard_normal(self, param_shape):
        return torch.normal(mean=0, std=1, size=param_shape, device=self.device)
