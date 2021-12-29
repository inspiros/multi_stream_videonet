from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from .non_local import (
    _MultiheadNonlocalNd,
    _MutualMultiheadNonlocalNd,
    _compute_conv_output_shape,
    _compute_output_padding
)

__all__ = [
    'DEQMultiheadNonlocal1d',
    'DEQMultiheadNonlocal2d',
    'DEQMultiheadNonlocal3d',
    'DEQMutualMultiheadNonlocal1d',
    'DEQMutualMultiheadNonlocal2d',
    'DEQMutualMultiheadNonlocal3d',
]


# noinspection PyMethodOverriding
class _DEQMultiheadNonlocalNd(_MultiheadNonlocalNd):

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, _ConvNd):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,
                z: torch.Tensor,
                x: torch.Tensor):
        batch_sz = z.size(0)
        in_dims = z.size()[2:]
        hidden_dims = _compute_conv_output_shape(input_shape=in_dims,
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 dilation=self.dilation)

        qkv = []
        for head_id in range(self.num_heads):
            q, k, v = self.embed_conv[head_id](z).flatten(2).chunk(3, dim=1)
            attn = torch.einsum('bcm,bcn->bmn', q, k).mul(self.scale).softmax(dim=-1)
            qkv.append(torch.einsum('bmn,bcn->bcm', attn, v))
        qkv = torch.cat(qkv, dim=1).view(batch_sz,
                                         self.hidden_channels * self.num_heads,
                                         *hidden_dims)

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


class DEQMultiheadNonlocal1d(_DEQMultiheadNonlocalNd):

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
        super(DEQMultiheadNonlocal1d, self).__init__(1,
                                                     in_channels,
                                                     hidden_channels,
                                                     num_heads,
                                                     kernel_size,
                                                     stride,
                                                     padding,
                                                     dilation,
                                                     scaled,
                                                     residual)


class DEQMultiheadNonlocal2d(_DEQMultiheadNonlocalNd):

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
        super(DEQMultiheadNonlocal2d, self).__init__(2,
                                                     in_channels,
                                                     hidden_channels,
                                                     num_heads,
                                                     kernel_size,
                                                     stride,
                                                     padding,
                                                     dilation,
                                                     scaled,
                                                     residual)


class DEQMultiheadNonlocal3d(_DEQMultiheadNonlocalNd):

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
        super(DEQMultiheadNonlocal3d, self).__init__(3,
                                                     in_channels,
                                                     hidden_channels,
                                                     num_heads,
                                                     kernel_size,
                                                     stride,
                                                     padding,
                                                     dilation,
                                                     scaled,
                                                     residual)


# noinspection PyMethodOverriding
class _DEQMutualMultiheadNonlocalNd(_MutualMultiheadNonlocalNd):

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, _ConvNd):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,
                zs: List[torch.Tensor],
                xs: List[torch.Tensor],
                ) -> List[torch.Tensor]:
        if len(zs) != self.num_streams:
            raise ValueError(f'Number of input streams must be {self.num_streams}; '
                             f'got {len(zs)}.')
        in_dims_list = [zs[stream_id].size() for stream_id in range(self.num_streams)]
        if not all(in_dims_list[stream_id - 1] == in_dims_list[stream_id]
                   for stream_id in range(1, self.num_streams)):
            raise ValueError(f'Different dimensions of streams are not supported; '
                             f'got {in_dims_list}.')

        qs, ks, vs = [], [], []
        for stream_id in range(self.num_streams):
            q_i, k_i, v_i = [], [], []
            for head_id in range(self.num_heads):
                q, k, v = self.embed_conv[stream_id](zs[stream_id]).flatten(2).chunk(3, dim=1)
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


class DEQMutualMultiheadNonlocal1d(_DEQMutualMultiheadNonlocalNd):

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
        super(DEQMutualMultiheadNonlocal1d, self).__init__(1,
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


class DEQMutualMultiheadNonlocal2d(_DEQMutualMultiheadNonlocalNd):

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
        super(DEQMutualMultiheadNonlocal2d, self).__init__(2,
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


class DEQMutualMultiheadNonlocal3d(_DEQMutualMultiheadNonlocalNd):

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
        super(DEQMutualMultiheadNonlocal3d, self).__init__(3,
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
