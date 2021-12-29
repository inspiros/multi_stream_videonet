from typing import Tuple, Optional, Callable, List, Type, Any, Union

import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url

from .fusion import WeightedFusionBlock
from .transformer import MutualMultiheadNonlocal3d
from .parallel_module_list import ParallelModuleList

from .r2plus1d import Conv3DSimple, Conv2Plus1D, Conv3DNoTemporal, BasicBlock, Bottleneck, BasicStem, R2Plus1dStem

__all__ = ['MultiStreamVideoResNet',
           'multi_stream_r3d_18',
           'multi_stream_mc3_18',
           'multi_stream_r2plus1d_18']

model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}


class MultiStreamVideoResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            conv_makers: List[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]],
            layers: List[int],
            stem: Callable[..., nn.Module],
            num_classes: int = 400,
            num_streams: int = 1,
            fusion_weights: bool = True,
            fusion_stage: int = 6,
            transfer_stages: List[int] = tuple(),
            zero_init_residual: bool = False,
    ) -> None:
        """Multi-stream Version of Generic resnet video generator.
        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): resnet building block
            conv_makers (List[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]]): generator
                function for each layer
            layers (List[int]): number of blocks per layer
            stem (Callable[..., nn.Module]): module specifying the ResNet stem.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            num_streams (int, optional): Number of input streams. Defaults to 1.
            fusion_weights (bool, optional): Apply learnable fusion weights. Defaults to True.
            fusion_stage (int, optional): Stage of fusion.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(MultiStreamVideoResNet, self).__init__()
        if fusion_stage > 6:
            raise ValueError(f'fusion_stage must be either 0 (after stem), 1 (after layer1), '
                             f'2 (after layer2), 3 (after layer3), 4 (after layer4), '
                             f'5 (after average pooling), or 6 (after fc/logit fusion). '
                             f'Got fusion_stage={fusion_stage}.')
        if any(transfer_stage > fusion_stage for transfer_stage in transfer_stages):
            raise ValueError(f'transfer_stages must be less or equal to fusion_stage={fusion_stage}. '
                             f'Got transfer_stages={transfer_stages}.')
        if any(transfer_stage >= 5 for transfer_stage in transfer_stages):
            raise ValueError(f'transfer_stages 5 and 6 (vector features) are currently not supported. '
                             f'Got transfer_stages={transfer_stages}.')

        self.inplanes = 64
        self.num_streams = num_streams
        self.fusion_stage = fusion_stage
        self.transfer_stages = transfer_stages

        # stem
        self.stem = ParallelModuleList([stem()] * self.num_streams)

        # layer1
        if self.fusion_stage >= 1:
            self.layer1 = ParallelModuleList(
                [self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)] * self.num_streams
            )
        else:
            self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)

        # layer2
        if self.fusion_stage >= 2:
            self.layer2 = ParallelModuleList(
                [self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)] * self.num_streams
            )
        else:
            self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)

        # layer3
        if self.fusion_stage >= 3:
            self.layer3 = ParallelModuleList(
                [self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)] * self.num_streams
            )
        else:
            self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)

        # layer4
        if self.fusion_stage >= 4:
            self.layer4 = ParallelModuleList(
                [self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)] * self.num_streams
            )
        else:
            self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        # global avarage pooling
        if self.fusion_stage >= 5:
            self.avgpool = ParallelModuleList(
                [nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                               nn.Flatten())] * self.num_streams)
        else:
            self.avgpool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                         nn.Flatten())

        # fc
        if self.fusion_stage >= 6:
            self.fc = ParallelModuleList(
                [nn.Linear(512 * block.expansion, num_classes)] * self.num_streams
            )
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # fuse and transfer
        stage_channels = {
            0: 64,
            1: 64,
            2: 128,
            3: 256,
            4: 512,
            5: 512,
            6: num_classes
        }
        self.transfer = nn.ModuleList([nn.Identity()] * 7)
        for transfer_stage in self.transfer_stages:
            self.transfer[transfer_stage] = \
                MutualMultiheadNonlocal3d(num_streams=self.num_streams,
                                          in_channels=stage_channels[transfer_stage],
                                          hidden_channels=stage_channels[transfer_stage] // 4,
                                          num_heads=1)
        self.fuse = nn.ModuleList([nn.Identity()] * 7)
        self.fuse[self.fusion_stage] = \
            WeightedFusionBlock(num_streams=self.num_streams,
                                in_channels=stage_channels[self.fusion_stage] if fusion_weights else None)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[union-attr, arg-type]

    def forward(self, x: List[Tensor]) -> Tensor:
        x = self.stem(x)
        x = self.transfer[0](x)
        x = self.fuse[0](x)

        x = self.layer1(x)
        x = self.transfer[1](x)
        x = self.fuse[1](x)

        x = self.layer2(x)
        x = self.transfer[2](x)
        x = self.fuse[2](x)

        x = self.layer3(x)
        x = self.transfer[3](x)
        x = self.fuse[3](x)

        x = self.layer4(x)
        x = self.transfer[4](x)
        x = self.fuse[4](x)

        x = self.avgpool(x)
        x = self.transfer[5](x)
        x = self.fuse[5](x)

        x = self.fc(x)
        x = self.transfer[6](x)
        x = self.fuse[6](x)

        return x

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

    def _initialize_weights(self) -> None:
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
