from collections import OrderedDict
from typing import List

from torch import nn, Tensor

from .multi_stream_model import MultiStreamVideoModel
from .._movinet_configs import *
from ..movinet import (
    Swish, CausalModule, TemporalAverage,
    ConvBlock3D, BasicBottleneck
)

__all__ = [
    'multi_stream_movinet_a0',
    'multi_stream_movinet_a1',
    'multi_stream_movinet_a2',
    'multi_stream_movinet_a3',
    'multi_stream_movinet_a4',
    'multi_stream_movinet_a5',
]


# noinspection DuplicatedCode
class MultiStreamMoViNet(MultiStreamVideoModel):
    def __init__(self,
                 cfg: 'MoViNetConfigNode',
                 causal: bool = True,
                 pretrained: bool = False,
                 num_classes: int = 600,
                 num_streams: int = 1,
                 weighted_fusion: bool = True,
                 fusion_stage: int = 6,
                 transfer_stages: List[int] = tuple(),
                 conv_type: str = '3d',
                 tf_like: bool = False
                 ) -> None:
        super(MultiStreamMoViNet, self).__init__(num_streams)
        """
        causal: causal mode
        pretrained: pretrained models
        If pretrained is True:
            num_classes is set to 600,
            conv_type is set to "3d" if causal is False,
                "2plus1d" if causal is True
            tf_like is set to True
        num_classes: number of classes for classifcation
        conv_type: type of convolution either 3d or 2plus1d
        tf_like: tf_like behaviour, basically same padding for convolutions
        """
        if pretrained:
            tf_like = True
            num_classes = 600
            conv_type = '2plus1d' if causal else '3d'

        norm_layer = nn.BatchNorm3d if conv_type == '3d' else nn.BatchNorm2d
        activation_layer = Swish if conv_type == '3d' else nn.Hardswish

        self.num_streams = num_streams
        self.fusion_stage = fusion_stage
        self.weighted_fusion = weighted_fusion
        self.transfer_stages = transfer_stages
        self.causal = causal

        # conv1
        self.conv1 = self._make_multi_stream_block(
            ConvBlock3D(
                in_planes=cfg.conv1.input_channels,
                out_planes=cfg.conv1.out_channels,
                kernel_size=cfg.conv1.kernel_size,
                stride=cfg.conv1.stride,
                padding=cfg.conv1.padding,
                causal=causal,
                conv_type=conv_type,
                tf_like=tf_like,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            ),
            out_channels=cfg.conv1.out_channels,
        )

        # blocks
        blocks_dict = OrderedDict()
        for i, block in enumerate(cfg.blocks):
            blocks_dict_i = OrderedDict()
            for j, basic_block in enumerate(block):
                blocks_dict_i[f'l{j}'] = self._make_multi_stream_block(
                    BasicBottleneck(
                        basic_block,
                        causal=causal,
                        conv_type=conv_type,
                        tf_like=tf_like,
                        norm_layer=norm_layer,
                        activation_layer=activation_layer
                    ),
                    no_transfer=True,
                    no_fusion=True,
                    keep_stage=True,
                )
            blocks_dict[f'b{i}'] = self._make_multi_stream_block(
                nn.Sequential(blocks_dict_i),
                out_channels=block[-1].out_channels,
                no_multi_stream=True,
            )
        self.blocks = nn.Sequential(blocks_dict)

        # conv7
        self.conv7 = self._make_multi_stream_block(
            ConvBlock3D(
                in_planes=cfg.conv7.input_channels,
                out_planes=cfg.conv7.out_channels,
                kernel_size=cfg.conv7.kernel_size,
                stride=cfg.conv7.stride,
                padding=cfg.conv7.padding,
                causal=causal,
                conv_type=conv_type,
                tf_like=tf_like,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            ),
            out_channels=cfg.conv7.out_channels,
        )

        # pool
        self.avg = self._make_multi_stream_block(
            TemporalAverage(self.causal),
            out_channels=cfg.conv7.out_channels,
            no_transfer=True,
        )

        # clf
        self.classifier = self._make_multi_stream_block(
            nn.Sequential(
                # dense9
                ConvBlock3D(cfg.conv7.out_channels,
                            cfg.dense9.hidden_dim,
                            kernel_size=(1, 1, 1),
                            tf_like=tf_like,
                            causal=causal,
                            conv_type=conv_type,
                            bias=True),
                Swish(),
                nn.Dropout(p=0.2, inplace=True),
                # dense10d
                ConvBlock3D(cfg.dense9.hidden_dim,
                            num_classes,
                            kernel_size=(1, 1, 1),
                            tf_like=tf_like,
                            causal=causal,
                            conv_type=conv_type,
                            bias=True),
            ),
            out_channels=num_classes,
            no_transfer=True,
        )

        self._check_stages()

        # load pre-trained model
        if pretrained:
            raise NotImplementedError
        else:
            self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv7(x)
        x = self.avg(x)
        x = self.classifier(x)
        x = x.flatten(1)

        return x

    @staticmethod
    def _clean_activation_buffers(m):
        if issubclass(type(m), CausalModule):
            m.reset_activation()

    def clean_activation_buffers(self) -> None:
        self.apply(self._clean_activation_buffers)


def multi_stream_movinet_a0(**kwargs):
    """Constructor for the MoViNet A0 network
    """
    return MultiStreamMoViNet(cfg=get_config('movinet_a0'), **kwargs)


def multi_stream_movinet_a1(**kwargs):
    """Constructor for the MoViNet A0 network
    """
    return MultiStreamMoViNet(cfg=get_config('movinet_a1'), **kwargs)


def multi_stream_movinet_a2(**kwargs):
    """Constructor for the MoViNet A0 network
    """
    return MultiStreamMoViNet(cfg=get_config('movinet_a2'), **kwargs)


def multi_stream_movinet_a3(**kwargs):
    """Constructor for the MoViNet A0 network
    """
    return MultiStreamMoViNet(cfg=get_config('movinet_a3'), **kwargs)


def multi_stream_movinet_a4(**kwargs):
    """Constructor for the MoViNet A0 network
    """
    return MultiStreamMoViNet(cfg=get_config('movinet_a4'), **kwargs)


def multi_stream_movinet_a5(**kwargs):
    """Constructor for the MoViNet A0 network
    """
    return MultiStreamMoViNet(cfg=get_config('movinet_a5'), **kwargs)
