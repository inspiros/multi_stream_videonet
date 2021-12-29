from typing import List

import torch
import torch.nn as nn

from .attention import MultiheadAttention, MutualMultiheadAttention

__all__ = [
    'DEQMultiheadAttention',
    'DEQMutualMultiheadAttention'
]


# noinspection PyMethodOverriding
class DEQMultiheadAttention(MultiheadAttention):

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,
                z: torch.Tensor,
                x: torch.Tensor
                ) -> torch.Tensor:
        qkv = []
        for head_id in range(self.num_heads):
            q, k, v = self.embed_fc[head_id](z).chunk(3, dim=1)
            attn = torch.einsum('bmd,bnd->bmn', q, k).mul(self.scale).softmax(dim=-1)
            qkv.append(torch.einsum('bmn,bnd->bmd', attn, v))
        qkv = torch.cat(qkv, dim=1)
        out = self.output_fc(qkv)
        if self.residual:
            out = out + x
        return out


# noinspection PyMethodOverriding
class DEQMutualMultiheadAttention(MutualMultiheadAttention):

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,
                zs: List[torch.Tensor],
                xs: List[torch.Tensor]
                ) -> List[torch.Tensor]:
        if len(zs) != self.num_streams:
            raise ValueError(f'Number of input streams must be {self.num_streams}. '
                             f'Got {len(zs)}.')
        shapes = [zs[stream_id].size() for stream_id in range(self.num_streams)]
        if not torch.tensor([shapes[stream_id - 1] == shapes[stream_id]
                             for stream_id in range(1, self.num_streams)]).all():
            raise ValueError(f'Different dimensions of streams are not supported. '
                             f'Got {shapes}.')

        qs, ks, vs = [], [], []
        for stream_id in range(self.num_streams):
            q_i, k_i, v_i = [], [], []
            for head_id in range(self.num_heads):
                q, k, v = self.embed_fc[stream_id](zs[stream_id]).chunk(3, dim=1)
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
                attn = torch.einsum('bmd,bnd->bmn', q, k).mul(self.scale).softmax(dim=-1)
                qkv.append(torch.einsum('bmn,bnd->bmd', attn, v))

            qkv = torch.cat(qkv, dim=1)
            out = self.output_fc[stream_id](qkv)
            if self.residual:
                out = out + xs[stream_id]
            outs.append(out)
        return outs
