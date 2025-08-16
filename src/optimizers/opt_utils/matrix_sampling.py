import torch
import math
import numpy as np
import random
import scipy.stats as sps
from collections import defaultdict

class MatrixSampler:
    
    def __init__(self, sampler_type, device='cuda'):
        
        self.sampler_type = sampler_type
        self.device = device
        if self.sampler_type == 'GS':                       # + +
            self.sampler = self._GS_matrix
        elif self.sampler_type == 'GS_v2':                  # + +
            self.sampler = self._GS_matrix_v2
        elif self.sampler_type == 'Householder_reflection': # + +
            self.sampler = self._householder_matrix
        elif self.sampler_type == 'Rotation':               # + + 
            self.sampler = self._rotation_matrix
        elif self.sampler_type == 'Reflection':             # + +
            self.sampler = self._reflection_matrix
        elif self.sampler_type == 'Random_baseline':        # + +
            self.sampler = self._random_baseline
        else:
            raise NotImplementedError(f"Sampling {self.sampler_type} is not implemented")
        
    def sample_single_matrix(self, param_shape, generator = None):
        assert len(param_shape) > 1, f"Sample only matrices, current shape: {param_shape}"
        n, m = param_shape
        if n > m:
            return self.sampler(n, generator=generator)[:, :m]
        return self.sampler(m, generator=generator)[:n, :]

    def sample(self, param_shapes):

        shape_to_names = defaultdict(list)
        for name, shape in param_shapes:
            shape_to_names[shape].append(name)

        E_dict = {}
        for (n, m), names in shape_to_names.items():
            
            k = min(n, m)
            S = self.Sigma(n, m)
            U = self.sampler(n)
            V = self.sampler(m)
            S_k = S[:k, :k]
            U_k = U[:, :k]
            V_k = V[:, :k]
            E_k = U_k @ S_k @ V_k.T
            # E = U @ S @ V.T
            for name in names:
                E_dict[name] = (E_k.clone(), U_k.clone(), S_k.clone(), V_k.clone())
                # E_dict[name] = (E.clone(), U.clone(), S.clone(), V.clone())
        return E_dict

    def Sigma(self, n, m, dtype=torch.float32):

        p = min(n, m)

        sigma = torch.zeros((n, m), device=self.device, dtype=dtype)
        sigma[torch.arange(p), torch.arange(p)] = torch.diag(self._rotation_matrix(p))

        return sigma

    def _householder_matrix(self, d, generator = None):
        
        u = torch.randn(d, device=self.device, generator=generator)
        H = torch.eye(d, device=self.device) - 2*(u*u.unsqueeze(1))/(u.norm()**2)

        return H


    def _rotation_matrix(self, d, num_rotations=None, generator = None):
        if num_rotations is None:
            num_rotations = d
        Q = torch.eye(d, device=self.device)
        for _ in range(num_rotations):
            i, j = torch.randint(0, d, (2,), device=self.device, generator=generator)
            while i == j:
                j = torch.randint(0, d, (1,), device=self.device, generator=generator)
                j = j.item()
            theta = torch.rand(1, device=self.device, generator=generator) * 2 * math.pi
            c = torch.cos(theta)
            s = torch.sin(theta)
            col_i = Q[:, i].clone()
            col_j = Q[:, j].clone()
            Q[:, i] = c * col_i - s * col_j
            Q[:, j] = s * col_i + c * col_j
        return Q


    def _reflection_matrix(self, d, generator = None):
        Q = torch.eye(d, device=self.device)
        idx = torch.randint(0, d - 1, (torch.randint(0,d-1, (1,), generator=generator), ))
        Q[idx, idx] = -1
        return Q


    def _random_baseline(self, d, generator = None):
        return torch.randn((d,d), device=self.device, generator=generator)

    def _GS_matrix(
        self,
        dim,
        num_blocks=10,
        use_PL=True,
        use_P=True,
        use_PR=True,
        generator=None,
    ):
        if generator is None:
            generator = torch.Generator(device=self.device)
        
        base = dim // num_blocks
        rem = dim % num_blocks
        blocks = [base + (1 if i < rem else 0) for i in range(num_blocks)]
        
        L = torch.zeros(dim, dim, device=self.device, dtype=torch.float32)
        R = torch.zeros(dim, dim, device=self.device, dtype=torch.float32)
        
        offset = 0
        for b in blocks:
            X_L = torch.randn(b, b, device=self.device, generator=generator)
            Q_L, R_L = torch.linalg.qr(X_L)
            sign_L = torch.sign(torch.diagonal(R_L, 0))
            Q_L = Q_L * sign_L.unsqueeze(0)
            if torch.det(Q_L) < 0:
                Q_L[:, 0] *= -1
            
            X_R = torch.randn(b, b, device=self.device, generator=generator)
            Q_R, R_R = torch.linalg.qr(X_R)
            sign_R = torch.sign(torch.diagonal(R_R, 0))
            Q_R = Q_R * sign_R.unsqueeze(0)
            if torch.det(Q_R) < 0:
                Q_R[:, 0] *= -1
            
            L[offset:offset+b, offset:offset+b] = Q_L
            R[offset:offset+b, offset:offset+b] = Q_R
            offset += b
        
        identity = torch.arange(dim, device=self.device)
        idx_PR = torch.randperm(dim, device=self.device, generator=generator) if use_PR else identity
        idx_P = torch.randperm(dim, device=self.device, generator=generator) if use_P else identity
        idx_PL = torch.randperm(dim, device=self.device, generator=generator) if use_PL else identity
        
        A = R[idx_P, :][:, idx_PR]
        A = L @ A
        A = A[idx_PL, :]
        
        return A
    
    def _GS_matrix_v2(
        self,
        dim,
        num_blocks=None,
        use_PL=True,
        use_P=True,
        use_PR=True,
        generator=None
    ):
        if generator is None:
            generator = torch.Generator(device=self.device)
        
        if num_blocks is None:
            num_blocks = max(1, int(np.sqrt(dim)))
        
        base = dim // num_blocks
        rem = dim % num_blocks
        blocks = [base + (1 if i < rem else 0) for i in range(num_blocks)]
        
        L = torch.zeros(dim, dim, device=self.device, dtype=torch.float32)
        R = torch.zeros(dim, dim, device=self.device, dtype=torch.float32)
        
        offset = 0
        for b in blocks:
            X_L = torch.randn(b, b, device=self.device, generator=generator)
            Q_L, R_L = torch.linalg.qr(X_L)
            sign_L = torch.sign(torch.diagonal(R_L, 0))
            Q_L = Q_L * sign_L.unsqueeze(0)
            if torch.det(Q_L) < 0:
                Q_L[:, 0] *= -1
            
            X_R = torch.randn(b, b, device=self.device, generator=generator)
            Q_R, R_R = torch.linalg.qr(X_R)
            sign_R = torch.sign(torch.diagonal(R_R, 0))
            Q_R = Q_R * sign_R.unsqueeze(0)
            if torch.det(Q_R) < 0:
                Q_R[:, 0] *= -1
            
            L[offset:offset+b, offset:offset+b] = Q_L
            R[offset:offset+b, offset:offset+b] = Q_R
            offset += b
        
        identity = torch.arange(dim, device=self.device)
        idx_PR = torch.randperm(dim, device=self.device, generator=generator) if use_PR else identity
        idx_P = torch.randperm(dim, device=self.device, generator=generator) if use_P else identity
        idx_PL = torch.randperm(dim, device=self.device, generator=generator) if use_PL else identity
        
        A = R[idx_P, :][:, idx_PR]
        A = L @ A
        A = A[idx_PL, :]
        
        return A
