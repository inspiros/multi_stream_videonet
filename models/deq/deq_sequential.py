import torch.nn as nn

__all__ = ['DEQSequential']


# noinspection PyMethodOverriding
class DEQSequential(nn.Sequential):

    def forward(self, zs, xs):
        for module in self:
            zs = module(zs, xs)
        return zs
