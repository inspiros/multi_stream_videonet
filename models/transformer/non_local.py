from typing import List, Tuple, Union, Optional

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple

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


class _MultiheadNonlocalNd(nn.Module):

    def __init__(self,
                 dimension: int,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_heads: int = 1,
                 kernel_size: Union[int, Tuple[int, ...]] = 1,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 scaled: bool = True,
                 residual: bool = True):
        super(_MultiheadNonlocalNd, self).__init__()
        _to_tuple = _ntuple(dimension)
        _embed_conv_module = _get_conv_module(dimension)
        _output_conv_module = _get_conv_module(dimension, transpose=True)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.num_heads = num_heads
        self.residual = residual
        self.scale = in_channels ** -0.5 if scaled else 1

        self.kernel_size = _to_tuple(kernel_size)
        self.stride = _to_tuple(stride)
        self.padding = _to_tuple(padding)
        self.dilation = _to_tuple(dilation)

        self.embed_conv = nn.ModuleList(
            [_embed_conv_module(in_channels,
                                self.hidden_channels * 3,
                                kernel_size=self.kernel_size,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation)
             for _ in range(self.num_heads)]
        )
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
        batch_sz = x.size(0)
        in_dims = x.size()[2:]
        hidden_dims = _compute_conv_output_shape(input_shape=in_dims,
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 dilation=self.dilation)

        qkv = []
        for head_id in range(self.num_heads):
            q, k, v = self.embed_conv[head_id](x).flatten(2).chunk(3, dim=1)
            attn = torch.einsum('bcm,bcn->bmn', q, k).mul(self.scale).softmax(dim=-1)
            qkv.append(torch.einsum('bmn,bcn->bcm', attn, v))
            print(head_id, q.shape, attn.shape)
            # attn = torch.bmm(q.transpose(1, 2), k)
            # qkv.append(torch.bmm(qk, v.transpose(1, 2)).transpose(1, 2))
        qkv = torch.cat(qkv, dim=1).view(batch_sz,
                                         self.hidden_channels * self.num_heads,
                                         *hidden_dims)

        print(qkv.shape)
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

    def __init__(self,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_heads: int = 1,
                 kernel_size: Union[int, Tuple[int, ...]] = 1,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 scaled: bool = True,
                 residual: bool = True):
        super(MultiheadNonlocal1d, self).__init__(1,
                                                  in_channels,
                                                  hidden_channels,
                                                  num_heads,
                                                  kernel_size,
                                                  stride,
                                                  padding,
                                                  dilation,
                                                  scaled,
                                                  residual)


class MultiheadNonlocal2d(_MultiheadNonlocalNd):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_heads: int = 1,
                 kernel_size: Union[int, Tuple[int, ...]] = 1,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 scaled: bool = True,
                 residual: bool = True):
        super(MultiheadNonlocal2d, self).__init__(2,
                                                  in_channels,
                                                  hidden_channels,
                                                  num_heads,
                                                  kernel_size,
                                                  stride,
                                                  padding,
                                                  dilation,
                                                  scaled,
                                                  residual)


class MultiheadNonlocal3d(_MultiheadNonlocalNd):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_heads: int = 1,
                 kernel_size: Union[int, Tuple[int, ...]] = 1,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 scaled: bool = True,
                 residual: bool = True):
        super(MultiheadNonlocal3d, self).__init__(3,
                                                  in_channels,
                                                  hidden_channels,
                                                  num_heads,
                                                  kernel_size,
                                                  stride,
                                                  padding,
                                                  dilation,
                                                  scaled,
                                                  residual)


class _MutualMultiheadNonlocalNd(nn.Module):

    def __init__(self,
                 dimension: int,
                 num_streams: int,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_heads: int = 1,
                 kernel_size: Union[int, Tuple[int, ...]] = 1,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 scaled: bool = True,
                 residual: bool = True):
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
        self.scale = in_channels ** -0.5 if scaled else 1

        self.kernel_size = _to_tuple(kernel_size)
        self.stride = _to_tuple(stride)
        self.padding = _to_tuple(padding)
        self.dilation = _to_tuple(dilation)

        self.embed_conv = nn.ModuleList(
            [_embed_conv_module(in_channels,
                                self.hidden_channels * 3,
                                kernel_size=self.kernel_size,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation)
             for _ in range(self.num_streams)]
        )
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

        qs, ks, vs = [], [], []
        for stream_id in range(self.num_streams):
            q_i, k_i, v_i = [], [], []
            for head_id in range(self.num_heads):
                q, k, v = self.embed_conv[stream_id](xs[stream_id]).flatten(2).chunk(3, dim=1)
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
                attn = torch.einsum('bcm,bcn->bmn', q, k).mul(self.scale).softmax(dim=-1)
                qkv.append(torch.einsum('bmn,bcn->bcm', attn, v))

            batch_sz = in_dims_list[stream_id][0]
            in_dims = in_dims_list[stream_id][2:]
            hidden_dims = _compute_conv_output_shape(input_shape=in_dims,
                                                     kernel_size=self.kernel_size,
                                                     stride=self.stride,
                                                     padding=self.padding,
                                                     dilation=self.dilation)

            qkv = torch.cat(qkv, dim=1).view(batch_sz,
                                             self.hidden_channels * self.num_heads,
                                             *hidden_dims)

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

    def __init__(self,
                 num_streams: int,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_heads: int = 1,
                 kernel_size: Union[int, Tuple[int, ...]] = 1,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 scaled: bool = True,
                 residual: bool = True):
        super(MutualMultiheadNonlocal1d, self).__init__(1,
                                                        num_streams,
                                                        in_channels,
                                                        hidden_channels,
                                                        num_heads,
                                                        kernel_size,
                                                        stride,
                                                        padding,
                                                        dilation,
                                                        scaled,
                                                        residual)


class MutualMultiheadNonlocal2d(_MutualMultiheadNonlocalNd):

    def __init__(self,
                 num_streams: int,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_heads: int = 1,
                 kernel_size: Union[int, Tuple[int, ...]] = 1,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 scaled: bool = True,
                 residual: bool = True):
        super(MutualMultiheadNonlocal2d, self).__init__(2,
                                                        num_streams,
                                                        in_channels,
                                                        hidden_channels,
                                                        num_heads,
                                                        kernel_size,
                                                        stride,
                                                        padding,
                                                        dilation,
                                                        scaled,
                                                        residual)


class MutualMultiheadNonlocal3d(_MutualMultiheadNonlocalNd):

    def __init__(self,
                 num_streams: int,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_heads: int = 1,
                 kernel_size: Union[int, Tuple[int, ...]] = 1,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 scaled: bool = True,
                 residual: bool = True):
        super(MutualMultiheadNonlocal3d, self).__init__(3,
                                                        num_streams,
                                                        in_channels,
                                                        hidden_channels,
                                                        num_heads,
                                                        kernel_size,
                                                        stride,
                                                        padding,
                                                        dilation,
                                                        scaled,
                                                        residual)
