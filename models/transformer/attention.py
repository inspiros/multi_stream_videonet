import torch
import torch.nn as nn

__all__ = [
    'none_attention',
    'softmax_attention',
    'nystrom_attention',
    'NoneAttention',
    'SoftmaxAttention',
    'NystromAttention',
]


# -----------------------
# Attention Functions
# -----------------------
def none_attention(q, k, v):
    return v


def softmax_attention(q, k, v, scaled=True, dim_last=True):
    """
    Perform the baseline Self-Attention presented in https://arxiv.org/abs/2102.03902.

    Args:
        q (Tensor): Querry tensor of shape [N, L, H, D].
        k (Tensor): Key tensor of shape [N, L, H, D].
        v (Tensor): Value tensor of shape [N, L, H, D].
        scaled (bool): If `True`, use scaled dot product by D^{-0.5}. Default to `True`.
        dim_last (bool): If `False`, in put dimension is [N, H, D, L] instead. Default to `True`.
    """
    if dim_last:
        D = q.size(-1)
        attn_eq, out_eq = 'bmhd,bnhd->bhmn', 'bhmn,bnhd->bmhd'
    else:
        D = q.size(-2)
        attn_eq, out_eq = 'bhdm,bhdn->bhmn', 'bhmn,bhdn->bhdm'

    scale = D ** -0.5 if scaled else 1
    attn = torch.einsum(attn_eq, q, k).mul_(scale).softmax(dim=-1)
    out = torch.einsum(out_eq, attn, v)
    return out.contiguous()


def _iterative_inv(mat, n_iter=6, method='inv_init_coeff_option'):
    assert mat.size(-1) == mat.size(-2)
    I = torch.eye(mat.size(-1), device=mat.device)
    K = mat

    if method == 'original':
        # The entries of K are positive and ||K||_{\inf} = 1 due to softmax
        # This original implementation is more conservative to compute coefficient of Z_0.
        V = 1 / torch.max(torch.sum(K, dim=-2)) * K.transpose(-1, -2)
    elif method == 'inv_init_coeff_option':
        # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0 (faster).
        V = 1 / torch.max(torch.sum(K, dim=-2), dim=-1).values[:, :, None, None] * K.transpose(-1, -2)
    else:
        raise ValueError(f'method {method} not supported.')

    for _ in range(n_iter):
        KV = torch.matmul(K, V)
        V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
    return V


def nystrom_attention(q, k, v,
                      scaled=True,
                      dim_last=True,
                      num_landmarks=None,
                      p_landmark=None,
                      num_iters=6,
                      method='inv_init_coeff_option'):
    """
    Perform the Nyström-Based Algorithm for Approximating Self-Attention presented in https://arxiv.org/abs/2102.03902.
    Code modified from https://github.com/mlpen/Nystromformer.

    Args:
        q (Tensor): Querry tensor of shape [N, L, H, D].
        k (Tensor): Key tensor of shape [N, L, H, D].
        v (Tensor): Value tensor of shape [N, L, H, D].
        scaled (bool): If `True`, use scaled dot product by D^{-0.5}. Default to `True`.
        dim_last (bool): If `False`, in put dimension is [N, H, D, L] instead. Default to `True`.
        num_landmarks (int): Number of landmarks. Default to None.
        p_landmark (float): Percentage of landmarks overriding num_landmarks. Default to None.
        num_iters (int): Number of iterative_inverse iterations. Default to 6.
        method (str): Method of initialization of interative_inverse. Default to 'inv_init_coeff_option'.
    """
    L_dim = -3 if dim_last else -1
    L = q.size(L_dim)
    if p_landmark is None:
        if num_landmarks is None:
            num_landmarks = L
        elif num_landmarks not in range(0, L + 1) or L % num_landmarks != 0:
            raise ValueError('num_landmarks must be divisible by seq_len, got '
                             f'seq_len={L} and num_landmarks={num_landmarks}.')
    else:
        if not 0 < p_landmark <= 1:
            raise ValueError('p_landmark must be in range (0, 1],'
                             f'got p_landmark={p_landmark}.')
        num_landmarks = L * p_landmark
        if num_landmarks % 1 != 0:
            raise ValueError('p_landmark must be divisible by seq_len, got '
                             f'seq_len={L} and p_landmarks={p_landmark}.')
        num_landmarks = int(num_landmarks)

    if num_landmarks == L:
        # default softmax attention
        out = softmax_attention(q, k, v, scaled=scaled)
    else:
        if dim_last:
            N, _, H, D = q.size()
            view_shape = (N, num_landmarks, L // num_landmarks, H, D)
            attn_eq, out_eq = 'bmhd,bnhd->bhmn', 'bhmn,bnhd->bmhd'
        else:
            N, H, D, _ = q.size()
            view_shape = (N, H, D, num_landmarks, L // num_landmarks)
            attn_eq, out_eq = 'bhdm,bhdn->bhmn', 'bhmn,bhdn->bhdm'

        # nystrom approximation
        if scaled:
            scale = D ** -.25
            q = q * scale
            k = k * scale

        q_landmarks = q.view(*view_shape).mean(dim=L_dim)
        k_landmarks = k.view(*view_shape).mean(dim=L_dim)

        kernel_1 = torch.einsum(attn_eq, q, k_landmarks).softmax(dim=-1)
        kernel_2 = torch.einsum(attn_eq, q_landmarks, k_landmarks).softmax(dim=-1)
        kernel_3 = torch.einsum(attn_eq, q_landmarks, k).softmax(dim=-1)
        out = torch.einsum(out_eq,
                           torch.matmul(kernel_1, _iterative_inv(kernel_2, n_iter=num_iters, method=method)),
                           torch.einsum(out_eq, kernel_3, v))
    return out.contiguous()


# -----------------------
# Attention Modules
# -----------------------
class NoneAttention(nn.Module):
    def forward(self, q, k, v):
        return none_attention(q, k, v)


class SoftmaxAttention(NoneAttention):
    def __init__(self,
                 scaled=True,
                 dim_last=True):
        super(SoftmaxAttention, self).__init__()
        self.scaled = scaled
        self.dim_last = dim_last

    def forward(self, q, k, v):
        return softmax_attention(q, k, v,
                                 scaled=self.scaled,
                                 dim_last=self.dim_last)


class NystromAttention(SoftmaxAttention):
    def __init__(self,
                 scaled=True,
                 dim_last=True,
                 num_landmarks=None,
                 p_landmark=None,
                 num_iters=6,
                 method='inv_init_coeff_option'):
        super(NystromAttention, self).__init__(scaled, dim_last)
        self.num_landmarks = num_landmarks
        self.p_landmark = p_landmark
        self.num_iters = num_iters
        self.method = method

    def forward(self, q, k, v):
        return nystrom_attention(q, k, v,
                                 scaled=self.scaled,
                                 dim_last=self.dim_last,
                                 num_landmarks=self.num_landmarks,
                                 p_landmark=self.p_landmark,
                                 num_iters=self.num_iters,
                                 method=self.method)


if __name__ == '__main__':
    torch.manual_seed(1)
    q = torch.randn(1, 8, 1, 16) * 10
    k = torch.randn_like(q) * 10
    v = torch.randn_like(q) * 10

    # fake heads
    q = q.repeat_interleave(2, dim=-2)
    k = k.repeat_interleave(2, dim=-2)
    v = v.repeat_interleave(2, dim=-2)

    dim_last = True
    if dim_last:
        head_dim = -2
    else:
        q = q.permute(0, 2, 3, 1).contiguous()
        k = k.permute(0, 2, 3, 1).contiguous()
        v = v.permute(0, 2, 3, 1).contiguous()
        head_dim = -3

    print('q,k,v', q.shape)

    y_sm = softmax_attention(q, k, v, dim_last=dim_last)
    print('y_softmax', y_sm.shape)
    assert torch.allclose(y_sm.select(head_dim, 0), y_sm.select(head_dim, 1))

    y_nys = nystrom_attention(q, k, v, p_landmark=0.5, dim_last=dim_last)
    print('y_nystrom', y_nys.shape)
    assert torch.allclose(y_nys.select(head_dim, 0), y_nys.select(head_dim, 1))

    diff = y_nys - y_sm
    print(diff.abs().mean())
