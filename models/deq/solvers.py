from collections import namedtuple
from typing import Callable, Union

import numpy as np
import torch

from ._tensor_utils import *

__all__ = ['anderson', 'broyden']

FixedPointSolverReturnType = namedtuple('FixedPointSolverReturnType',
                                        ['result', 'lowest', 'nstep', 'abs_trace', 'rel_trace'])


def anderson(f: Callable,
             x0: torch.Tensor,
             collate_fn: Union[Callable, str] = 'default',
             decollate_fn: Union[Callable, str] = 'default',
             m: int = 6,
             lam: float = 1e-4,
             max_iters: int = 50,
             tol: float = 1e-3,
             eps: float = 1e-5,
             stop_mode: str = 'rel',
             beta: float = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    if collate_fn == 'default':
        collate_fn = get_default_collate_fn()
    elif collate_fn is None:
        collate_fn = identity
    if decollate_fn == 'default':
        decollate_fn = get_default_decollate_fn(x0)
    elif decollate_fn is None:
        decollate_fn = identity
    if stop_mode not in ['abs', 'rel']:
        raise ValueError
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'

    X0, F0 = collate_fn(x0), collate_fn(f(x0))
    X1, F1 = F0, collate_fn(f(decollate_fn(F0)))

    options = dict(dtype=X0.dtype, device=X0.device)
    batch_sz = X0.size(0)
    dim = prod(X0.size()[1:])

    X = torch.zeros(batch_sz, m, dim, **options)
    F = torch.zeros(batch_sz, m, dim, **options)
    X[:, 0], F[:, 0] = X0, F0
    X[:, 1], F[:, 1] = X1, F1
    del X0, X1, F0, F1

    H = torch.zeros(batch_sz, m + 1, m + 1, **options)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(batch_sz, m + 1, 1, **options)
    y[:, 0] = 1

    lowest_xest = x0
    trace_dict = dict(abs=[], rel=[])
    lowest_dict = dict(abs=np.inf, rel=np.inf)
    lowest_step_dict = dict(abs=0, rel=0)

    for k in range(2, max_iters):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, **options)[None]
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]  # (batch_sz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = collate_fn(f(decollate_fn(X[:, k % m])))
        gx = decollate_fn(F[:, k % m] - X[:, k % m])
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (F[:, k % m].norm().item() + eps)

        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)

        for diff, mode in zip([rel_diff, abs_diff], ['rel', 'abs']):
            if diff < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest = decollate_fn(X[:, k % m].clone().detach())
                    lowest_gx = gx.clone().detach()
                lowest_dict[mode] = diff
                lowest_step_dict[mode] = k

        if trace_dict[stop_mode][-1] < tol:
            for _ in range(max_iters - 1 - k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break

    return FixedPointSolverReturnType(result=lowest_xest,
                                      lowest=lowest_dict[stop_mode],
                                      nstep=lowest_step_dict[stop_mode],
                                      abs_trace=trace_dict['abs'],
                                      rel_trace=trace_dict['rel'])


def original_anderson(f, x0, m=6, lam=1e-4, max_iters=50, eps=1e-3, stop_mode='rel', beta=1.0, **kwargs):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, L = x0.shape
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'

    X = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape_as(x0)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    trace_dict = {'abs': [],
                  'rel': []}
    lowest_dict = {'abs': 1e8,
                   'rel': 1e8}
    lowest_step_dict = {'abs': 0,
                        'rel': 0}

    for k in range(2, max_iters):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
        gx = (F[:, k % m] - X[:, k % m]).view_as(x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + F[:, k % m].norm().item())
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)

        for mode in ['rel', 'abs']:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest = X[:, k % m].view_as(x0).clone().detach()
                    lowest_gx = gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k

        if trace_dict[stop_mode][-1] < eps:
            for _ in range(max_iters - 1 - k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break

    out = {"result": lowest_xest,
           "lowest": lowest_dict[stop_mode],
           "nstep": lowest_step_dict[stop_mode],
           "prot_break": False,
           "abs_trace": trace_dict['abs'],
           "rel_trace": trace_dict['rel'],
           "eps": eps,
           "max_iters": max_iters}
    return out


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)  # First do an update with step size 1
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -derphi0 * alpha0 ** 2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:  # we are assuming alpha>0 is a descent direction
        factor = alpha0 ** 2 * alpha1 ** 2 * (alpha1 - alpha0)
        a = alpha0 ** 2 * (phi_a1 - phi0 - derphi0 * alpha1) - \
            alpha1 ** 2 * (phi_a0 - phi0 - derphi0 * alpha0)
        a = a / factor
        b = -alpha0 ** 3 * (phi_a1 - phi0 - derphi0 * alpha1) + \
            alpha1 ** 3 * (phi_a0 - phi0 - derphi0 * alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b ** 2 - 3 * a * derphi0))) / (3.0 * a)
        phi_a2 = phi(alpha2)
        ite += 1

        if phi_a2 <= phi0 + c1 * alpha2 * derphi0:
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2 / alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1, ite


def line_search(update, x0, g0, g, nstep=0, on=True):
    """
    `update` is the propsoed direction of update.
    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0) ** 2]
    s_norm = torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]  # If the step size is so small... just return something
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = torch.norm(g0_new) ** 2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new

    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite


def rmatvec(part_Us, part_VTs, x):
    # Compute x^T(-I + UV^T)
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', max_iters)
    # part_VTs: (N, max_iters, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum('bij, bijd -> bd', x, part_Us)  # (N, max_iters)
    return -x + torch.einsum('bd, bdij -> bij', xTU, part_VTs)  # (N, 2d, L'), but should really be (N, 1, (2d*L'))


def matvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', max_iters)
    # part_VTs: (N, max_iters, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum('bdij, bij -> bd', part_VTs, x)  # (N, max_iters)
    return -x + torch.einsum('bijd, bd -> bij', part_Us, VTx)  # (N, 2d, L'), but should really be (N, (2d*L'), 1)


def broyden(f, x0, max_iters, eps=1e-3, stop_mode="rel", ls=False, **kwargs):
    bsz, total_hsize, seq_len = x0.size()
    g = lambda y: f(y) - y
    dev = x0.device
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'

    x_est = x0  # (bsz, 2d, L')
    gx = g(x_est)  # (bsz, 2d, L')
    k = 0
    tnstep = 0

    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, total_hsize, seq_len, max_iters).to(
        dev)  # One can also use an L-BFGS scheme to further reduce memory
    VTs = torch.zeros(bsz, max_iters, total_hsize, seq_len).to(dev)
    update = -matvec(Us[:, :, :, :k], VTs[:, :k], gx)  # Formally should be -torch.matmul(inv_jacobian (-I), gx)
    prot_break = False

    # To be used in protective breaks
    protect_thres = (1e6 if stop_mode == "abs" else 1e3) * seq_len
    new_objective = 1e8

    trace_dict = {'abs': [],
                  'rel': []}
    lowest_dict = {'abs': 1e8,
                   'rel': 1e8}
    lowest_step_dict = {'abs': 0,
                        'rel': 0}
    lowest_xest, lowest_gx = x_est, gx

    for k in range(max_iters):
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=k, on=ls)
        k += 1
        tnstep += (ite + 1)
        abs_diff = torch.norm(gx).item()
        rel_diff = abs_diff / (torch.norm(gx + x_est).item() + 1e-9)
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)
        for mode in ['rel', 'abs']:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k

        new_objective = diff_dict[stop_mode]
        if new_objective < eps: break
        if (new_objective < 3 * eps and k > 30 and
                np.max(trace_dict[stop_mode][-30:]) / np.min(trace_dict[stop_mode][-30:]) < 1.3):
            # if there's hardly been any progress in the last 30 steps
            break
        if new_objective > trace_dict[stop_mode][0] * protect_thres:
            prot_break = True
            break

        part_Us, part_VTs = Us[:, :, :, :k - 1], VTs[:, :k - 1]
        vT = rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / torch.einsum('bij, bij -> b', vT, delta_gx)[:, None, None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:, k - 1] = vT
        Us[:, :, :, k - 1] = u
        update = -matvec(Us[:, :, :, :k], VTs[:, :k], gx)

    # Fill everything up to the max_iters length
    for _ in range(max_iters + 1 - len(trace_dict[stop_mode])):
        trace_dict[stop_mode].append(lowest_dict[stop_mode])
        trace_dict[alternative_mode].append(lowest_dict[alternative_mode])

    return {"result": lowest_xest,
            "lowest": lowest_dict[stop_mode],
            "nstep": lowest_step_dict[stop_mode],
            "prot_break": prot_break,
            "abs_trace": trace_dict['abs'],
            "rel_trace": trace_dict['rel'],
            "eps": eps,
            "max_iters": max_iters}
