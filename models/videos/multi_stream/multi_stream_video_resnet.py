from typing import Callable, List, Type, Any, Union

import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url

from .multi_stream_model import MultiStreamVideoModel
from ..video_resnet import (
    Conv3DSimple, Conv2Plus1D, Conv3DNoTemporal,
    BasicBlock, Bottleneck, BasicStem, R2Plus1dStem
)

__all__ = [
    'MultiStreamVideoResNet',
    'multi_stream_r3d_18',
    'multi_stream_mc3_18',
    'multi_stream_r2plus1d_18'
]

model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}


# noinspection DuplicatedCode
class MultiStreamVideoResNet(MultiStreamVideoModel):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            conv_makers: List[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]],
            layers: List[int],
            stem: Callable[..., nn.Module],
            num_classes: int = 400,
            num_streams: int = 1,
            weighted_fusion: bool = True,
            fusion_stage: int = 6,
            transfer_stages: List[int] = tuple(),
            zero_init_residual: bool = False,
    ):
        """Multi-stream Version of Generic resnet video generator.
        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): resnet building block
            conv_makers (List[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]]): generator
                function for each layer
            layers (List[int]): number of blocks per layer
            stem (Callable[..., nn.Module]): module specifying the ResNet stem.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            num_streams (int, optional): Number of input streams. Defaults to 1.
            weighted_fusion (bool, optional): Apply learnable fusion weights. Defaults to True.
            fusion_stage (int, optional): Stage of fusion.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(MultiStreamVideoResNet, self).__init__(num_streams)

        self.inplanes = 64
        self.fusion_stage = fusion_stage
        self.weighted_fusion = weighted_fusion
        self.transfer_stages = transfer_stages

        # stem
        self.stem = self._make_multi_stream_block(
            stem(),
            out_channels=64,
        )

        # layer1
        self.layer1 = self._make_multi_stream_block(
            self._make_layer(block, conv_makers[0], 64, layers[0], stride=1),
            out_channels=64,
        )

        # layer2
        self.layer2 = self._make_multi_stream_block(
            self._make_layer(block, conv_makers[1], 128, layers[1], stride=2),
            out_channels=128,
        )

        # layer3
        self.layer3 = self._make_multi_stream_block(
            self._make_layer(block, conv_makers[2], 256, layers[2], stride=2),
            out_channels=256,
        )

        # layer4
        self.layer4 = self._make_multi_stream_block(
            self._make_layer(block, conv_makers[3], 512, layers[3], stride=2),
            out_channels=512,
        )

        # global avarage pooling
        self.avgpool = self._make_multi_stream_block(
            nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten()),
            out_channels=512,
            no_transfer=True,
        )

        # fc
        self.fc = self._make_multi_stream_block(
            nn.Linear(512 * block.expansion, num_classes),
            out_channels=num_classes,
            no_transfer=True,
        )

        self._check_stages()

        # init weights
        self._initialize_weights(zero_init_residual)

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            conv_builder: Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]],
            planes: int,
            blocks: int,
            stride: int = 1
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = [block(self.inplanes, planes, conv_builder, stride, downsample)]

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[union-attr, arg-type]

    def forward(self, x: List[Tensor]) -> Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x


def _multi_stream_video_resnet(arch: str,
                               pretrained: bool = False,
                               progress: bool = True,
                               **kwargs: Any) -> MultiStreamVideoResNet:
    model = MultiStreamVideoResNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        multistream_state_dict = dict()

        def update_multi_stream_params(old_k, state):
            new_k = old_k[:old_k.find('.')] + '.{}.' + old_k[old_k.find('.') + 1:]
            for i in range(model.num_streams):
                multistream_state_dict[new_k.format(i)] = state

        for k, v in state_dict.items():
            sub_module_name = k[:k.find('.')]
            if sub_module_name == 'stem' and model.fusion_stage > 1:
                update_multi_stream_params(k, v)
            elif sub_module_name == 'layer1' and model.fusion_stage > 2:
                update_multi_stream_params(k, v)
            elif sub_module_name == 'layer2' and model.fusion_stage > 3:
                update_multi_stream_params(k, v)
            elif sub_module_name == 'layer3' and model.fusion_stage > 4:
                update_multi_stream_params(k, v)
            elif sub_module_name == 'layer4' and model.fusion_stage > 5:
                update_multi_stream_params(k, v)
            elif sub_module_name == 'fc' and model.fusion_stage > 6:
                update_multi_stream_params(k, v)
            else:
                multistream_state_dict[k] = v
        if 'num_classes' in kwargs and kwargs['num_classes'] != 400:
            if 'fc.weight' in multistream_state_dict:
                multistream_state_dict.pop('fc.weight')
            if 'fc.bias' in multistream_state_dict:
                multistream_state_dict.pop('fc.bias')
        model.load_state_dict(multistream_state_dict, strict=False)
    return model


def multi_stream_r3d_18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MultiStreamVideoResNet:
    """Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr
    Returns:
        nn.Module: R3D-18 network
    """
    return _multi_stream_video_resnet('r3d_18',
                                      pretrained, progress,
                                      block=BasicBlock,
                                      conv_makers=[Conv3DSimple] * 4,
                                      layers=[2, 2, 2, 2],
                                      stem=BasicStem, **kwargs)


def multi_stream_mc3_18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MultiStreamVideoResNet:
    """Constructor for 18 layer Mixed Convolution network as in
    https://arxiv.org/abs/1711.11248
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr
    Returns:
        nn.Module: MC3 Network definition
    """
    return _multi_stream_video_resnet('mc3_18',
                                      pretrained, progress,
                                      block=BasicBlock,
                                      conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,  # type: ignore[list-item]
                                      layers=[2, 2, 2, 2],
                                      stem=BasicStem, **kwargs)


def multi_stream_r2plus1d_18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MultiStreamVideoResNet:
    """Constructor for the multi-stream 18 layer deep R(2+1)D network as in
    https://arxiv.org/abs/1711.11248
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr
    Returns:
        nn.Module: R(2+1)D-18 network
    """
    return _multi_stream_video_resnet('r2plus1d_18',
                                      pretrained, progress,
                                      block=BasicBlock,
                                      conv_makers=[Conv2Plus1D] * 4,
                                      layers=[2, 2, 2, 2],
                                      stem=R2Plus1dStem, **kwargs)
