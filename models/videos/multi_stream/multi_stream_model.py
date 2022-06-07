from collections import OrderedDict

import torch.nn as nn

from .fusion import FusionBlock
from .parallel_modules import ParallelModuleList
from ...transformer.non_local import MutualMultiheadNonlocal3d

__all__ = ['MultiStreamVideoModel']


class MultiStreamVideoModel(nn.Module):
    def __init__(self, num_streams):
        super(MultiStreamVideoModel, self).__init__()
        self.num_streams = num_streams
        self.__current_stage = 0

    def _make_multi_stream_block(self,
                                 module,
                                 out_channels=None,
                                 no_multi_stream=False,
                                 no_transfer=False,
                                 no_fusion=False,
                                 keep_stage=False):
        stage = self.__current_stage
        multi_stream = stage <= self.fusion_stage and not no_multi_stream
        transfer = stage in self.transfer_stages and not no_transfer
        fusion = stage == self.fusion_stage and not no_fusion

        layers = OrderedDict(
            module=ParallelModuleList([module] * self.num_streams) if multi_stream else module,
            transfer=MutualMultiheadNonlocal3d(num_streams=self.num_streams,
                                               in_channels=out_channels,
                                               hidden_channels=out_channels // 4,
                                               attn_type='nystrom',
                                               p_landmark=1 / 16)
            if transfer and out_channels is not None else nn.Identity(),
            fusion=FusionBlock(num_streams=self.num_streams,
                               in_channels=out_channels,
                               weighted=self.weighted_fusion)
            if fusion else nn.Identity(),
        )
        m = nn.Sequential(layers)
        m.stage = stage
        if not keep_stage:
            self.__current_stage += 1
        return m

    def _check_stages(self):
        final_stage = self.__current_stage
        assert 0 <= self.fusion_stage <= final_stage
        assert all(0 <= _ <= final_stage for _ in self.transfer_stages)
