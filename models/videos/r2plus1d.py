"""
This is the r2.5d implementation.

References
----------
[1] https://arxiv.org/abs/1711.11248
"""
import torch
import torch.nn as nn

__all__ = [
    'r2plus1d_10',
    'r2plus1d_16',
    'r2plus1d_18',
    'r2plus1d_26',
    'r2plus1d_34',
    'r2plus1d_50',
    'r2plus1d_101',
    'r2plus1d_152',
    'r2plus1d_200',
]


class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):

        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """
    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class R2plus1D(nn.Module):

    def __init__(self, layers,
                 block,
                 num_classes=400,
                 zero_init_residual=False):
        """R(2+1)D.

        Args:
            layers (List[int]): number of blocks per layer
            block (nn.Module): block type, `BasicBlock` or `Bottleneck`
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(R2plus1D, self).__init__()
        self.inplanes = 64

        self.stem = R2Plus1dStem()

        self.layer1 = self._make_layer(block, Conv2Plus1D, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, Conv2Plus1D, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, Conv2Plus1D, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, Conv2Plus1D, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        self._initialize_weights(zero_init_residual)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x_feat = x.flatten(1)
        x = self.fc(x_feat)

        return x, x_feat

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
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

    def _initialize_weights(self, zero_init_residual=False):
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
            elif zero_init_residual and isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def load_state_dict(self, state_dict, strict=False):
        current_state_dict = self.state_dict()
        for key in list(state_dict.keys()):
            if key in current_state_dict.keys() and state_dict[key].shape != current_state_dict[key].shape:
                print(f"[Warning] Key {key} has incompatible shape of {state_dict[key].shape}, "
                      f"expecting {current_state_dict[key].shape}.")
                state_dict.pop(key)
        super().load_state_dict(state_dict, strict)


def r2plus1d_10(**kwargs):
    """Constructor for the 10 layer R(2+1)D network
    """
    return R2plus1D(block=BasicBlock, layers=[1, 1, 1, 1], **kwargs)


def r2plus1d_16(**kwargs):
    """Constructor for the 16 layer R(2+1)D network
    """
    return R2plus1D(block=BasicBlock, layers=[2, 2, 2, 1], **kwargs)


def r2plus1d_18(**kwargs):
    """Constructor for the 18 layer R(2+1)D network
    """
    return R2plus1D(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)


def r2plus1d_26(**kwargs):
    """Construct 26 layer R(2+1)D network
    """
    return R2plus1D(block=Bottleneck, layers=[2, 2, 2, 2], **kwargs)


def r2plus1d_34(**kwargs):
    """Construct 34 layer R(2+1)D network
    """
    return R2plus1D(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)


def r2plus1d_50(**kwargs):
    """Construct 50 layer R(2+1)D network
    """
    return R2plus1D(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)


def r2plus1d_101(**kwargs):
    """Construct 101 layer R(2+1)D network
    """
    return R2plus1D(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)


def r2plus1d_152(**kwargs):
    """Construct 152 layer R(2+1)D network
    """
    return R2plus1D(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)


def r2plus1d_200(**kwargs):
    """Construct 200 layer R(2+1)D network
    """
    return R2plus1D(block=Bottleneck, layers=[3, 24, 36, 3], **kwargs)


if __name__ == '__main__':
    model = r2plus1d_18(num_classes=12)

    inputs = torch.randn(5, 3, 16, 112, 112)
    output = model(inputs)
    print(output.shape)
