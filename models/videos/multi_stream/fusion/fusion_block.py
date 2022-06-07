from typing import List, Optional

import torch
import torch.nn as nn

__all__ = ['FusionBlock']


class FusionBlock(nn.Module):

    def __init__(self,
                 num_streams: int,
                 in_channels: Optional[int] = None,
                 weighted=True):
        super(FusionBlock, self).__init__()
        self.num_streams = num_streams
        self.in_channels = in_channels
        self.weighted = weighted and in_channels is not None
        if self.weighted:
            self.fusion_weights = torch.nn.Parameter(torch.rand(self.num_streams, in_channels))
        else:
            self.register_parameter('fusion_weights', None)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        for stream_id in range(self.num_streams):
            # xs[stream_id] = xs[stream_id] / xs[stream_id].norm(dim=1, keepdim=True)
            if self.fusion_weights is not None:
                xs[stream_id] = torch.mul(xs[stream_id],
                                          self.fusion_weights[stream_id].view((self.in_channels,
                                                                               *([1] * (xs[stream_id].ndim - 2))))
                                          )
        return sum(xs)
