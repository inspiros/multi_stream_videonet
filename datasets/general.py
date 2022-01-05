from torch.utils.data import Dataset

from .utils.video_sampler import *

__all__ = ['MultiStreamVideoDataset']


class MultiStreamVideoDataset(Dataset):
    def __init__(self,
                 subsets,
                 synchronize_samplers=False):
        if len(set(len(_) for _ in subsets)) > 1:
            raise ValueError('All subsets must have identical length')
        if synchronize_samplers and not all(hasattr(subset, 'sampler') for subset in subsets):
            raise ValueError('All subsets must hold a Sampler for synchronization')
        self.subsets = subsets
        self.n_streams = len(self.subsets)

    def __len__(self):
        return len(self.subsets[0])

    def __getitem__(self, item):
        data, label = [], []
        with synchronize_state([subset.sampler for subset in self.subsets]):
            for subset in self.subsets:
                X, y = subset[item]
                data.append(X)
                label.append(y)
        if len(set(label)) > 1:
            raise RuntimeError('Label not consistent.')
        label = label[0]
        return (*data, label)
