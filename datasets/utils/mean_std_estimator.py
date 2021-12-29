import torch
from tqdm import tqdm

__all__ = ['compute_mean_std']


@torch.no_grad()
def _round(tensor, decimals=0):
    return (tensor.to(torch.float64) * 10 ** decimals).round() / (10 ** decimals)


def compute_mean_std(dataset, n_channels=3, device='cpu', verbose=True):
    device = torch.device(device)

    count = 0
    s1 = s2 = torch.zeros(n_channels, device=device)
    pbar = enumerate(dataset)
    if verbose:
        pbar = tqdm(pbar, total=len(dataset), desc='[compute_mean_std]')
    for sample_id, X in pbar:
        if isinstance(X, tuple):
            X = X[0]
        is_tube = X.ndim == 4
        val = X.to(dtype=torch.float32, device=device).mean([-1, -2])
        if not is_tube:
            count += 1
            s1 += val
            s2 += val.pow(2)
        else:
            count += X.size(1)
            s1 += val.sum(1)
            s2 += val.pow(2).sum(1)
        if verbose:
            if sample_id % 5 == 0 or sample_id == len(dataset) - 1:
                desc = f'[compute_mean_std iter {{iter}}/{len(dataset)}] '
                desc += f'running_mean={_round(s1 / count, decimals=3).tolist()}, '
                desc += f'running_std={_round(torch.sqrt((s2 - s1.pow(2) / count) / (count - 1)), decimals=3).tolist()}'
            pbar.set_description(desc.format(iter=sample_id + 1))

    mean = s1 / count
    std = torch.sqrt((s2 - s1.pow(2) / count) / (count - 1))
    return mean, std
