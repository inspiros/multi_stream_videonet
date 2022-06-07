from typing import List, Optional

import torch
from torch import nn

from .attention import *

__all__ = [
    'MultiheadAttention',
    'MutualMultiheadAttention'
]


# noinspection DuplicatedCode
class MultiheadAttention(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 num_heads: int = 1,
                 attn_type: str = 'softmax',
                 scaled: bool = True,
                 residual: bool = True,
                 **attn_kwargs):
        super(MultiheadAttention, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features if hidden_features is not None else in_features
        self.num_heads = num_heads
        self.residual = residual

        if attn_type == 'none':
            self.attn = NoneAttention()
        elif attn_type == 'softmax':
            self.attn = SoftmaxAttention(scaled=scaled)
        elif attn_type == 'nystrom':
            self.attn = NystromAttention(scaled=scaled, **attn_kwargs)
        else:
            raise ValueError(f'attn_type {attn_type} not supported.')

        self.embed_fc = nn.Linear(in_features, self.hidden_features * self.num_heads * 3)
        self.output_fc = nn.Linear(self.hidden_features * self.num_heads, in_features)

        # init weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, L, D = x.size()

        embed = self.embed_fc(x).chunk(3, dim=-1)
        q = embed[0].view(N, L, self.num_heads, self.hidden_features)
        k = embed[1].view(N, L, self.num_heads, self.hidden_features)
        v = embed[2].view(N, L, self.num_heads, self.hidden_features)
        qkv = self.attn(q, k, v).view(N, L, self.hidden_features)
        out = self.output_fc(qkv)
        if self.residual:
            out = out + x
        return out


# noinspection DuplicatedCode
class MutualMultiheadAttention(nn.Module):

    def __init__(self,
                 num_streams: int,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 num_heads: int = 1,
                 attn_type: str = 'softmax',
                 scaled: bool = True,
                 residual: bool = True,
                 **attn_kwargs):
        super(MutualMultiheadAttention, self).__init__()
        if num_streams < 2:
            raise ValueError(f'num_streams must be greater or equal to 2. Got {num_streams}.')

        self.in_features = in_features
        self.hidden_features = hidden_features if hidden_features is not None else in_features
        self.num_streams = num_streams
        self.num_heads = num_heads
        self.residual = residual

        if attn_type == 'none':
            self.attn = NoneAttention()
        elif attn_type == 'softmax':
            self.attn = SoftmaxAttention(scaled=scaled)
        elif attn_type == 'nystrom':
            self.attn = NystromAttention(scaled=scaled, **attn_kwargs)
        else:
            raise ValueError(f'attn_type {attn_type} not supported.')

        self.embed_fc = nn.ModuleList([
            nn.Linear(in_features, self.hidden_features * self.num_heads * 3)
            for _ in range(self.num_streams)
        ])
        self.output_fc = nn.ModuleList([
            nn.Linear(self.hidden_features * self.num_heads, in_features)
            for _ in range(self.num_streams)
        ])

        # init weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(xs) != self.num_streams:
            raise ValueError(f'Number of input streams must be {self.num_streams}. '
                             f'Got {len(xs)}.')
        in_dims_list = [xs[stream_id].size() for stream_id in range(self.num_streams)]
        if not all(in_dims_list[stream_id - 1] == in_dims_list[stream_id]
                   for stream_id in range(1, self.num_streams)):
            raise ValueError(f'Different dimensions of streams are not supported. '
                             f'Got {in_dims_list}.')
        N, L, D = in_dims_list[0]

        qs, ks, vs = [], [], []
        for stream_id in range(self.num_streams):
            embed = self.embed_fc[stream_id](xs[stream_id]).chunk(3, dim=-1)
            qs.append(embed[0].view(N, L, self.num_heads, self.hidden_features))
            ks.append(embed[1].view(N, L, self.num_heads, self.hidden_features))
            vs.append(embed[2].view(N, L, self.num_heads, self.hidden_features))

        outs = []
        for stream_id in range(self.num_streams):
            q = sum(qs[other_stream_id] for other_stream_id in range(self.num_streams)
                    if other_stream_id != stream_id)
            k = ks[stream_id]
            v = vs[stream_id]
            qkv = self.attn(q, k, v).view(N, L, self.hidden_features)

            out = self.output_fc[stream_id](qkv)
            if self.residual:
                out = out + xs[stream_id]
            outs.append(out)
        return outs
