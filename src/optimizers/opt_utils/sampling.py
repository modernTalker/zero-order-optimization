from collections import defaultdict
import inspect
import math
import os
import re
import random
import shutil
import sys
import scipy.stats as sps
from scipy.stats import ortho_group
import numpy as np
import torch

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
        

def generate_rotation_matrix(d, num_rotations=None, device='cpu'):
    if num_rotations is None:
        num_rotations = d
    Q = torch.eye(d, device=device)
    for _ in range(num_rotations):
        i, j = torch.randint(0, d, (2,), device=device)
        while i == j:
            j = torch.randint(0, d, (1,), device=device)
            j = j.item()
        theta = torch.rand(1, device=device) * 2 * math.pi
        c = torch.cos(theta)
        s = torch.sin(theta)
        col_i = Q[:, i].clone()
        col_j = Q[:, j].clone()
        Q[:, i] = c * col_i - s * col_j
        Q[:, j] = s * col_i + c * col_j
    return Q

def generate_orthogonal_matrix(d, num_rotations=None, device='cpu'):
    Q = generate_rotation_matrix(d, num_rotations, device=device)
    if torch.rand(1, device=device) < 0.5:
        Q[0, :] = -Q[0, :]
    return Q

def generate_reflection_matrix(d, device='cpu'):
    Q = torch.eye(d, device=device)
    idx = random.randint(0, d - 1)
    Q[idx, idx] = -1
    return Q

def generate_semi_diagonal(n, m, distribution=sps.norm, device='cpu', dtype=torch.float32):
    p = min(n, m)

    sigma_np = distribution.rvs(size=p)
    
    sigma = torch.tensor(sigma_np, device=device, dtype=dtype)

    Sigma = torch.zeros((n, m), device=device, dtype=dtype)
    Sigma[torch.arange(p), torch.arange(p)] = sigma

    return Sigma

def create_random_matrix(n, m, device='cpu'):
    U = generate_rotation_matrix(n, device=device)
    V = generate_rotation_matrix(m, device=device)
    
    S = generate_semi_diagonal(n, m, device=device)
    
    return U @ S @ V.T, U, V, S

def torch_ortho_rvs(dim: int, device="cpu", dtype=torch.float32):
    z = torch.randn(dim, dim, device=device, dtype=dtype)

    q, r = torch.linalg.qr(z)

    d = torch.diagonal(r, 0)
    ph = d.sign()
    q *= ph.unsqueeze(0)

    return q

def generate_micola_matrix(
    dim,
    num_blocks: int = 10,
    use_PL: bool = True,
    use_P:  bool = True,
    use_PR: bool = True,
    device='cpu',
):
    base = dim // num_blocks
    rem  = dim %  num_blocks
    blocks = [base + (1 if i < rem else 0) for i in range(num_blocks)]
    block_sizes_L = blocks
    block_sizes_R = blocks


    if block_sizes_L:
        bsL = block_sizes_L
        maxL = max(bsL)
        L_blocks = []
        for b in bsL:
            X = torch.randn(b, b, device=device)
            Qb, Rb = torch.linalg.qr(X)
            sign = torch.sign(torch.diagonal(Rb, 0))
            Qb *= sign
            if torch.det(Qb) < 0:
                Qb[:, 0] *= -1
            L_blocks.append(Qb)
        L = torch.block_diag(*L_blocks).to(device)
    else:
        L = torch.eye(dim, device=device)

    if block_sizes_R:
        bsR = block_sizes_R
        R_blocks = []
        for b in bsR:
            X = torch.randn(b, b, device=device)
            Qb, Rb = torch.linalg.qr(X)
            sign = torch.sign(torch.diagonal(Rb, 0))
            Qb *= sign
            if torch.det(Qb) < 0:
                Qb[:, 0] *= -1
            R_blocks.append(Qb)
        R = torch.block_diag(*R_blocks).to(device)
    else:
        R = torch.eye(dim, device=device)

    idx_PL = torch.randperm(dim, device=device) if use_PL else torch.arange(dim, device=device)
    idx_P  = torch.randperm(dim, device=device) if use_P  else torch.arange(dim, device=device)
    idx_PR = torch.randperm(dim, device=device) if use_PR else torch.arange(dim, device=device)

    M1 = R[:, idx_PR]
    M2 = M1[idx_P, :]
    M3 = L @ M2
    A  = M3[idx_PL, :]

    return A

def generate_reflection_matrix(d, device='cpu'):
    Q = torch.eye(d, device=device)
    i = torch.randint(0, d, (1,), device=device).item()
    Q[i, i] = -1
    return Q

def generate_rotation_via_householders(d, device='cpu'):
    v1 = torch.randn(d, device=device)
    v1 /= v1.norm()

    v2 = torch.randn(d, device=device)
    v2 -= (v1 * v2).sum() * v1
    v2 /= v2.norm()

    I = torch.eye(d, device=device)
    H1 = I - 2.0 * v1.unsqueeze(1) @ v1.unsqueeze(0)
    H2 = I - 2.0 * v2.unsqueeze(1) @ v2.unsqueeze(0)
    Q = H2 @ H1
    return Q

def generate_housholder_and_reflection_matrix(d, device='cpu'):
    Q = generate_rotation_via_householders(d, device=device)
    if torch.rand(1, device=device) < 0.5:
        Q[0, :] = -Q[0, :]
    return Q

######### FAST ONE ###########

def create_random_matrix_fast_big_matrix(param_shapes, device='cpu'):
    max_n = max(n for _, (n, m) in param_shapes)
    max_m = max(m for _, (n, m) in param_shapes)

    # U_big = generate_orthogonal_matrix(max_n, device=device)  107 sec / iter
    # V_big = generate_orthogonal_matrix(max_m, device=device)


    # U_big = generate_reflection_matrix(max_n, device=device)  0.1 sec / iter
    # V_big = generate_reflection_matrix(max_m, device=device)

    # U_big = torch.FloatTensor(ortho_group.rvs(max_n)).to(device)  8 sec / iter
    # V_big = torch.FloatTensor(ortho_group.rvs(max_m)).to(device)


    U_big = torch_ortho_rvs(max_n, device=device)  # 0.2 sec / iter ----> GREAT
    V_big = torch_ortho_rvs(max_n, device=device)

    E_dict = {}
    for name, (n, m) in param_shapes:
        k = min(n, m)
        U_slice = U_big[:n, :k]      # n x k
        V_slice = V_big[:m, :k]      # m x k
        # E = U @ I_nm @ V.T
        E = U_slice @ V_slice.T
        E_dict[name] = (E, U_slice, V_slice)
    return E_dict

def create_random_matrix_fast_per_param(param_shapes, device='cpu'):

    # U_big = generate_orthogonal_matrix(max_n, device=device)  107 sec / iter
    # V_big = generate_orthogonal_matrix(max_m, device=device)


    # U_big = generate_reflection_matrix(max_n, device=device)  0.1 sec / iter
    # V_big = generate_reflection_matrix(max_m, device=device)

    # U_big = torch.FloatTensor(ortho_group.rvs(max_n)).to(device)  8 sec / iter
    # V_big = torch.FloatTensor(ortho_group.rvs(max_m)).to(device)


    # U_big = torch_ortho_rvs(max_n, device=device)  # 0.2 sec / iter ----> GREAT
    # V_big = torch_ortho_rvs(max_n, device=device)

    E_dict = {}
    for name, (n, m) in param_shapes:
        k = min(n, m)

        U = torch_ortho_rvs(n, device=device)
        V = torch_ortho_rvs(m, device=device)

        U_slice = U[:, :k]  # (n x k)
        V_slice = V[:, :k]  # (m x k)

        E = U_slice @ V_slice.T  # (n x m)
        E_dict[name] = (E, U_slice, V_slice)

    return E_dict

def create_random_matrix_fast(param_shapes, device='cpu'):
    shape_to_names = defaultdict(list)
    for name, shape in param_shapes:
        shape_to_names[shape].append(name)

    E_dict = {}
    for (n, m), names in shape_to_names.items():
        k = min(n, m)
        # U = torch_ortho_rvs(n, device=device)
        # V = torch_ortho_rvs(m, device=device)
        U = generate_micola_matrix(n, device=device)
        V = generate_micola_matrix(m, device=device)
        U_k = U[:, :k]
        V_k = V[:, :k]
        E = U_k @ V_k.T
        for name in names:
            E_dict[name] = (E.clone(), U_k.clone(), V_k.clone())
    return E_dict


############################

def generate_orthogonal_approx(G, device='cpu'):
    assert len(G.shape) == 2, "Input must be a 2D tensor"
    m, n = G.shape
    
    # U_e m x m
    U_e = generate_orthogonal_matrix(m, device=device)
    
    #V_e n x n
    V_e = generate_orthogonal_matrix(n, device=device)
    
    approx = U_e @ V_e.T
    
    return approx

# Semiorthogonal setup

def generate_semi_rotation_matrix(m, n, num_rotations=None, device='cpu'):
    if num_rotations is None:
        num_rotations = min(m, n)

    k = min(m, n)
    V = torch.zeros((m, n), device=device)
    for i in range(min(m, k)):
        V[i, i] = 1.0

    for _ in range(num_rotations):
        i, j = torch.randint(0, k, (2,), device=device)
        while i == j:
            j = torch.randint(0, k, (1,), device=device)
            j = j.item()

        theta = torch.rand(1, device=device) * 2 * math.pi
        c = torch.cos(theta)
        s = torch.sin(theta)

        col_i = V[:, i].clone()
        col_j = V[:, j].clone()
        V[:, i] = c * col_i - s * col_j
        V[:, j] = s * col_i + c * col_j

    return V

def generate_semi_orthogonal_matrix(m, n, num_rotations=None, device='cpu'):
    V = generate_semi_rotation_matrix(m, n, num_rotations, device)
    if torch.rand(1, device=device) < 0.5:
        V[0, :] = -V[0, :]
    return V

def generate_semi_orthogonal_approx(G, device='cpu'):
    assert len(G.shape) == 2, "Input must be a 2D tensor"
    m, n = G.shape
    V_e = generate_semi_orthogonal_matrix(m, n, device=device)
    return V_e

###################

def sample_ortho_approx(shape, device='cpu'):
    assert len(shape) == 2, "Input dimension must be 2D"
    m, n = shape
    p = max(m, n)
    E = torch_ortho_rvs(p, device=device)
    return E[:m, :n]
