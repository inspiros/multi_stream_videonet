import functools
import math
import operator
import sys
from typing import List, Tuple, Union, Optional

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple

from .attention import *

__all__ = ['MultiheadNonlocal1d',
           'MultiheadNonlocal2d',
           'MultiheadNonlocal3d',
           'MutualMultiheadNonlocal1d',
           'MutualMultiheadNonlocal2d',
           'MutualMultiheadNonlocal3d',
           ]


def _get_conv_module(dimension, transpose=False):
    if dimension == 1:
        return nn.Conv1d if not transpose else nn.ConvTranspose1d
    elif dimension == 2:
        return nn.Conv2d if not transpose else nn.ConvTranspose2d
    elif dimension == 3:
        return nn.Conv3d if not transpose else nn.ConvTranspose3d
    raise ValueError(f'Only supports 1, 2, and 3-D; got dimension={dimension}.')


def _compute_conv_output_shape(input_shape: Tuple[int, ...],
                               kernel_size: Tuple[int, ...],
                               stride: Tuple[int, ...],
                               padding: Tuple[int, ...],
                               dilation: Tuple[int, ...],
                               ) -> Tuple[int, ...]:
    return tuple((i + 2 * p - (d * (k - 1) + 1)) // s + 1 for i, k, s, p, d in
                 zip(input_shape, kernel_size, stride, padding, dilation))


def _compute_conv_transpose_output_shape(input_shape: Tuple[int, ...],
                                         kernel_size: Tuple[int, ...],
                                         stride: Tuple[int, ...],
                                         padding: Tuple[int, ...],
                                         dilation: Tuple[int, ...],
                                         output_padding: Tuple[int, ...],
                                         ) -> Tuple[int, ...]:
    return tuple((i - 1) * s - 2 * p + d * (k - 1) + 1 + op for i, k, s, p, d, op in
                 zip(input_shape, kernel_size, stride, padding, dilation, output_padding))


def _compute_output_padding(input_shape: Tuple[int, ...],
                            output_shape: Tuple[int, ...],
                            kernel_size: Tuple[int, ...],
                            stride: Tuple[int, ...],
                            padding: Tuple[int, ...],
                            dilation: Tuple[int, ...],
                            ) -> Tuple[int, ...]:
    return tuple(o - (i - 1) * s + 2 * p - d * (k - 1) - 1 for i, o, k, s, p, d in
                 zip(input_shape, output_shape, kernel_size, stride, padding, dilation))


# noinspection DuplicatedCode
class _MultiheadNonlocalNd(nn.Module):

    def __init__(self,
                 dimension: int,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_heads: int = 1,
                 attn_type: str = 'softmax',
                 kernel_size: Union[int, Tuple[int, ...]] = 1,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 scaled: bool = True,
                 residual: bool = True,
                 **attn_kwargs):
        super(_MultiheadNonlocalNd, self).__init__()
        _to_tuple = _ntuple(dimension)
        _embed_conv_module = _get_conv_module(dimension)
        _output_conv_module = _get_conv_module(dimension, transpose=True)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.num_heads = num_heads
        self.residual = residual

        if attn_type == 'none':
            self.attn = NoneAttention()
        elif attn_type == 'softmax':
            self.attn = SoftmaxAttention(scaled=scaled, dim_last=False)
        elif attn_type == 'nystrom':
            self.attn = NystromAttention(scaled=scaled, dim_last=False, **attn_kwargs)
        else:
            raise ValueError(f'attn_type {attn_type} not supported.')

        self.kernel_size = _to_tuple(kernel_size)
        self.stride = _to_tuple(stride)
        self.padding = _to_tuple(padding)
        self.dilation = _to_tuple(dilation)

        self.embed_conv = _embed_conv_module(in_channels,
                                             self.hidden_channels * self.num_heads * 3,
                                             kernel_size=self.kernel_size,
                                             stride=self.stride,
                                             padding=self.padding,
                                             dilation=self.dilation)
        self.output_conv = _output_conv_module(self.hidden_channels * self.num_heads,
                                               in_channels,
                                               kernel_size=self.kernel_size,
                                               stride=self.stride,
                                               padding=self.padding,
                                               dilation=self.dilation)

        # init weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, _ConvNd):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C = x.size()[:2]
        in_dims = x.size()[2:]
        hidden_dims = _compute_conv_output_shape(input_shape=in_dims,
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 dilation=self.dilation)

        embed = self.embed_conv(x).chunk(3, dim=1)
        q = embed[0].flatten(2).view(N, self.num_heads, self.hidden_channels, -1)
        k = embed[1].flatten(2).view(N, self.num_heads, self.hidden_channels, -1)
        v = embed[2].flatten(2).view(N, self.num_heads, self.hidden_channels, -1)
        qkv = self.attn(q, k, v).view(N, self.num_heads * self.hidden_channels, *hidden_dims)

        self.output_conv.output_padding = _compute_output_padding(input_shape=hidden_dims,
                                                                  output_shape=in_dims,
                                                                  kernel_size=self.kernel_size,
                                                                  stride=self.stride,
                                                                  padding=self.padding,
                                                                  dilation=self.dilation)
        out = self.output_conv(qkv)
        if self.residual:
            out = out + x
        return out


class MultiheadNonlocal1d(_MultiheadNonlocalNd):
    def __init__(self, *args, **kwargs):
        super(MultiheadNonlocal1d, self).__init__(1, *args, **kwargs)


class MultiheadNonlocal2d(_MultiheadNonlocalNd):
    def __init__(self, *args, **kwargs):
        super(MultiheadNonlocal2d, self).__init__(2, *args, **kwargs)


class MultiheadNonlocal3d(_MultiheadNonlocalNd):
    def __init__(self, *args, **kwargs):
        super(MultiheadNonlocal3d, self).__init__(3, *args, **kwargs)


# noinspection DuplicatedCode
class _MutualMultiheadNonlocalNd(nn.Module):

    def __init__(self,
                 dimension: int,
                 num_streams: int,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_heads: int = 1,
                 attn_type: str = 'softmax',
                 kernel_size: Union[int, Tuple[int, ...]] = 1,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 scaled: bool = True,
                 residual: bool = True,
                 **attn_kwargs):
        super(_MutualMultiheadNonlocalNd, self).__init__()
        if num_streams < 2:
            raise ValueError(f'num_streams must be greater or equal to 2. Got {num_streams}.')
        _to_tuple = _ntuple(dimension)
        _embed_conv_module = _get_conv_module(dimension)
        _output_conv_module = _get_conv_module(dimension, transpose=True)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.num_streams = num_streams
        self.num_heads = num_heads
        self.residual = residual

        if attn_type == 'none':
            self.attn = NoneAttention()
        elif attn_type == 'softmax':
            self.attn = SoftmaxAttention(scaled=scaled, dim_last=False)
        elif attn_type == 'nystrom':
            self.attn = NystromAttention(scaled=scaled, dim_last=False, **attn_kwargs)
        else:
            raise ValueError(f'attn_type {attn_type} not supported.')

        self.kernel_size = _to_tuple(kernel_size)
        self.stride = _to_tuple(stride)
        self.padding = _to_tuple(padding)
        self.dilation = _to_tuple(dilation)

        self.embed_conv = nn.ModuleList([
            _embed_conv_module(in_channels,
                               self.hidden_channels * self.num_heads * 3,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding,
                               dilation=self.dilation)
            for _ in range(self.num_streams)
        ])
        self.output_conv = nn.ModuleList(
            [_output_conv_module(self.hidden_channels * self.num_heads,
                                 in_channels,
                                 kernel_size=self.kernel_size,
                                 stride=self.stride,
                                 padding=self.padding,
                                 dilation=self.dilation)
             for _ in range(self.num_streams)]
        )

        # init weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, _ConvNd):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(xs) != self.num_streams:
            raise ValueError(f'Number of input streams must be {self.num_streams}; '
                             f'got {len(xs)}.')
        in_dims_list = [xs[stream_id].size() for stream_id in range(self.num_streams)]
        if not all(in_dims_list[stream_id - 1] == in_dims_list[stream_id]
                   for stream_id in range(1, self.num_streams)):
            raise ValueError(f'Different dimensions of streams are not supported; '
                             f'got {in_dims_list}.')
        N, C = in_dims_list[0][:2]
        in_dims = in_dims_list[0][2:]
        hidden_dims = _compute_conv_output_shape(input_shape=in_dims,
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 dilation=self.dilation)

        qs, ks, vs = [], [], []
        for stream_id in range(self.num_streams):
            embed = self.embed_conv[stream_id](xs[stream_id]).chunk(3, dim=1)
            qs.append(embed[0].flatten(2).view(N, self.num_heads, self.hidden_channels, -1))
            ks.append(embed[1].flatten(2).view(N, self.num_heads, self.hidden_channels, -1))
            vs.append(embed[2].flatten(2).view(N, self.num_heads, self.hidden_channels, -1))

        outs = []
        for stream_id in range(self.num_streams):
            q = sum(qs[other_stream_id] for other_stream_id in range(self.num_streams)
                    if other_stream_id != stream_id)
            k = ks[stream_id]
            v = vs[stream_id]
            qkv = self.attn(q, k, v).view(N, self.num_heads * self.hidden_channels, *hidden_dims)

            self.output_conv[stream_id].output_padding = _compute_output_padding(input_shape=hidden_dims,
                                                                                 output_shape=in_dims,
                                                                                 kernel_size=self.kernel_size,
                                                                                 stride=self.stride,
                                                                                 padding=self.padding,
                                                                                 dilation=self.dilation)
            out = self.output_conv[stream_id](qkv)
            if self.residual:
                out = out + xs[stream_id]
            outs.append(out)
        return outs


class MutualMultiheadNonlocal1d(_MutualMultiheadNonlocalNd):
    def __init__(self, *args, **kwargs):
        super(MutualMultiheadNonlocal1d, self).__init__(1, *args, **kwargs)


class MutualMultiheadNonlocal2d(_MutualMultiheadNonlocalNd):
    def __init__(self, *args, **kwargs):
        super(MutualMultiheadNonlocal2d, self).__init__(2, *args, **kwargs)


class MutualMultiheadNonlocal3d(_MutualMultiheadNonlocalNd):
    def __init__(self, *args, **kwargs):
        super(MutualMultiheadNonlocal3d, self).__init__(3, *args, **kwargs)
