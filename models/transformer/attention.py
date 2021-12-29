from typing import List, Optional

import torch
from torch import nn

__all__ = ['MultiheadAttention',
           'MutualMultiheadAttention']


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiheadAttention(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_heads: int = 1,
                 scaled: bool = True,
                 residual: bool = True):
        super(MultiheadAttention, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.num_heads = num_heads
        self.residual = residual
        self.scale = in_channels ** -0.5 if scaled else 1

        self.embed_fc = nn.ModuleList(
            [nn.Linear(in_channels, self.hidden_channels * 3)
             for _ in range(self.num_heads)]
        )
        self.output_fc = nn.Linear(self.hidden_channels * self.num_heads, in_channels)

        # init weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = []
        for head_id in range(self.num_heads):
            q, k, v = self.embed_fc[head_id](x).chunk(3, dim=1)
            attn = torch.einsum('bmd,bnd->bmn', q, k).mul(self.scale).softmax(dim=-1)
            qkv.append(torch.einsum('bmn,bnd->bmd', attn, v))
            # attn = torch.bmm(q.transpose(1, 2), k)
            # qkv.append(torch.bmm(qk, v.transpose(1, 2)).transpose(1, 2))
        qkv = torch.cat(qkv, dim=1)
        out = self.output_fc(qkv)
        if self.residual:
            out = out + x
        return out


class MutualMultiheadAttention(nn.Module):

    def __init__(self,
                 num_streams: int,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_heads: int = 1,
                 scaled: bool = True,
                 residual: bool = True):
        super(MutualMultiheadAttention, self).__init__()
        if num_streams < 2:
            raise ValueError(f'num_streams must be greater or equal to 2. Got {num_streams}.')

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.num_streams = num_streams
        self.num_heads = num_heads
        self.residual = residual
        self.scale = in_channels ** -0.5 if scaled else 1

        self.embed_fc = nn.ModuleList(
            [nn.Linear(in_channels, self.hidden_channels * 3)
             for _ in range(self.num_streams)]
        )
        self.output_fc = nn.ModuleList(
            [nn.Linear(self.hidden_channels * self.num_heads, in_channels)
             for _ in range(self.num_streams)]
        )

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
        shapes = [xs[stream_id].size() for stream_id in range(self.num_streams)]
        if not torch.tensor([shapes[stream_id - 1] == shapes[stream_id]
                             for stream_id in range(1, self.num_streams)]).all():
            raise ValueError(f'Different dimensions of streams are not supported. '
                             f'Got {shapes}.')

        qs, ks, vs = [], [], []
        for stream_id in range(self.num_streams):
            q_i, k_i, v_i = [], [], []
            for head_id in range(self.num_heads):
                q, k, v = self.embed_fc[stream_id](xs[stream_id]).chunk(3, dim=1)
                q_i.append(q)
                k_i.append(k)
                v_i.append(v)
            qs.append(q_i)
            ks.append(k_i)
            vs.append(v_i)

        outs = []
        for stream_id in range(self.num_streams):
            qkv = []
            for head_id in range(self.num_heads):
                q = sum(qs[other_stream_id][head_id] for other_stream_id in range(self.num_streams)
                        if other_stream_id != stream_id)
                k = ks[stream_id][head_id]
                v = vs[stream_id][head_id]
                print(q.shape, k.shape, v.shape)
                attn = torch.einsum('bmd,bnd->bmn', q, k).mul(self.scale).softmax(dim=-1)
                qkv.append(torch.einsum('bmn,bnd->bmd', attn, v))

            qkv = torch.cat(qkv, dim=1)
            out = self.output_fc[stream_id](qkv)
            if self.residual:
                out = out + xs[stream_id]
            outs.append(out)
        return outs
