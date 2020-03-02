
"""Timm compatible BigLittleNets
TODO:integrate BigLittleModule and BigLittleModuleX
TODO:integrate _make_layer
TODO:proper parameter passing
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .helpers import load_pretrained
from .layers import SelectAdaptivePool2d, DropBlock2d, DropPath, AvgPool2dSame, create_attn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


__all__ = ['ResNet', 'BasicBlock', 'Bottleneck']  # model_registry will add each entrypoint fn to this


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {}


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    __constants__ = ['se', 'downsample']  # for pre 1.4 torchscript compat
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act3(x)

        return x

class BigLittleModule(nn.Module):
    def __init__(self, block, in_channels, out_channels, blocks, alpha, beta, stride,act_layer=nn.ReLU,norm_layer=nn.BatchNorm2d,
                 attn_layer=None, drop_block=None, drop_path=None):
        super(BigLittleModule, self).__init__()

        self.act1 = act_layer(inplace=True)
        self.big = self._make_layer(block, in_channels, out_channels, blocks - 1, 2, last_relu=False)
        self.little = self._make_layer(block, in_channels, out_channels // alpha, max(1, blocks // beta - 1))
        self.little_e = nn.Sequential(
            nn.Conv2d(out_channels // alpha, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels))

        self.fusion = self._make_layer(block, out_channels, out_channels, 1, stride=stride)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, norm_layer=nn.BatchNorm2d, last_relu=True):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(norm_layer(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)
        layers = []
        if blocks == 1:
            layers.append(block(inplanes, planes, stride=stride, downsample=downsample))
        else:
            layers.append(block(inplanes, planes, stride, downsample))
            for i in range(1, blocks):
                layers.append(block(planes, planes,
                                    last_relu=last_relu if i == blocks - 1 else True))

        return nn.Sequential(*layers)

    def forward(self, x):
        big = self.big(x)
        little = self.little(x)
        little = self.little_e(little)
        big = nn.functional.interpolate(big, little.shape[2:])
        out = self.relu(big + little)
        out = self.fusion(out)

        return out

class BigLittleModuleX(nn.Module):
    def __init__(self, block, in_channels, out_channels, blocks, basewidth, cardinality, alpha, beta, stride,
                 act_layer=nn.ReLU,norm_layer=nn.BatchNorm2d, attn_layer=None, drop_block=None, drop_path=None):
        super(BigLittleModuleX, self).__init__()

        self.act1 = act_layer(inplace=True)
        self.big = self._make_layer(block, in_channels, out_channels, blocks - 1,
                                    basewidth, cardinality, 2, last_relu=False)
        self.little = self._make_layer(block, in_channels, out_channels // alpha,
                                       max(1, blocks // beta - 1), basewidth * alpha, cardinality // alpha)
        self.little_e = nn.Sequential(
            nn.Conv2d(out_channels // alpha, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels))

        self.fusion = self._make_layer(block, out_channels, out_channels, 1, basewidth, cardinality, stride=stride)

    def _make_layer(self, block, inplanes, planes, blocks, basewidth, cardinality, stride=1, norm_layer=nn.BatchNorm2d, last_relu=True):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, padding=0, stride=1, bias=False))
            downsample.append(norm_layer(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)

        layers = []
        if blocks == 1:
            layers.append(block(inplanes, planes, basewidth, cardinality, stride=stride, downsample=downsample))
        else:
            layers.append(block(inplanes, planes, basewidth, cardinality, stride, downsample))
            for i in range(1, blocks):
                layers.append(block(planes, planes, basewidth, cardinality,
                                    last_relu=last_relu if i == blocks - 1 else True))

        return nn.Sequential(*layers)

    def forward(self, x):
        big = self.big(x)
        little = self.little(x)
        little = self.little_e(little)
        big = nn.functional.interpolate(big, little.shape[2:])
        out = self.relu(big + little)
        out = self.fusion(out)

        return out



def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


class BigLittleResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int
        Numbers of layers in each block
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    cardinality : int, default 1
        Number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64
        Factor determining bottleneck channels. `planes * base_width / 64 * cardinality`
    stem_width : int, default 64
        Number of channels in stem convolutions
    stem_type : str, default ''
        The type of stem:
          * '', default - a single 7x7 conv with a width of stem_width
          * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
          * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width//4 * 6, stem_width * 2
          * 'deep_tiered_narrow' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
    block_reduce_first: int, default 1
        Reduction factor for first convolution output width of residual blocks,
        1 for all archs except senets, where 2
    down_kernel_size: int, default 1
        Kernel size of residual block downsampling path, 1x1 for most archs, 3x3 for senets
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    output_stride : int, default 32
        Set the output stride of the network, 32, 16, or 8. Typically used in segmentation.
    act_layer : class, activation layer
    norm_layer : class, normalization layer
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    """
    def __init__(self, block, layers, alpha, beta, num_classes=1000, in_chans=3,
                 cardinality=1, base_width=64, stem_width=64, stem_type='',
                 block_reduce_first=1, down_kernel_size=1, avg_down=False, output_stride=32,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, drop_rate=0.0, drop_path_rate=0.,
                 drop_block_rate=0., global_pool='avg', zero_init_last_bn=True, block_args=None):
        block_args = block_args or dict()
        self.num_classes = num_classes
        deep_stem = 'deep' in stem_type
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.cardinality = cardinality
        self.base_width = base_width
        self.drop_rate = drop_rate
        self.expansion = block.expansion
        super(BigLittleResNet, self).__init__()

        # Stem
        if deep_stem:
            stem_chs_1 = stem_chs_2 = stem_width
            if 'tiered' in stem_type:
                stem_chs_1 = 3 * (stem_width // 4)
                stem_chs_2 = stem_width if 'narrow' in stem_type else 6 * (stem_width // 4)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs_1, 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs_1),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs_1, stem_chs_2, 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs_2),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs_2, self.inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.act1 = act_layer(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)



        # Feature Blocks
        dp = DropPath(drop_path_rate) if drop_path_rate else None
        db_3 = DropBlock2d(drop_block_rate, 7, 0.25) if drop_block_rate else None
        db_4 = DropBlock2d(drop_block_rate, 7, 1.00) if drop_block_rate else None
        channels, strides, dilations = [64, 128, 256, 512], [1, 2, 2, 2], [1] * 4
        #BigLittle Stem
        self.b_conv0 = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_b0 = norm_layer(channels[0])
        self.l_conv0 = nn.Conv2d(channels[0], channels[0] // alpha,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_l0 = norm_layer(channels[0] // alpha)
        self.l_conv1 = nn.Conv2d(channels[0] // alpha, channels[0] //
                                 alpha, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_l1 = norm_layer(channels[0] // alpha)
        self.l_conv2 = nn.Conv2d(channels[0] // alpha, channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_l2 = norm_layer(channels[0])
        self.bl_init = nn.Conv2d(channels[0], channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_bl_init = norm_layer(channels[0])

        if output_stride == 16:
            strides[3] = 1
            dilations[3] = 2
        elif output_stride == 8:
            strides[2:4] = [1, 1]
            dilations[2:4] = [2, 4]
        else:
            assert output_stride == 32
        layer_args = list(zip(channels, layers, strides, dilations))
        layer_kwargs = dict(
            reduce_first=block_reduce_first, act_layer=act_layer, norm_layer=norm_layer,
            avg_down=avg_down, down_kernel_size=down_kernel_size, drop_path=dp, **block_args)
        self.layer1 = self._make_layer(block, *layer_args[0], **layer_kwargs)
        #self.layer1 = self.BigLittleModule(block, *layer_args[0], **layer_kwargs)
        self.layer2 = self._make_layer(block, *layer_args[1], **layer_kwargs)
        #self.layer2 = self.BigLittleModule(block, *layer_args[1], **layer_kwargs)
        self.layer3 = self._make_layer(block, drop_block=db_3, *layer_args[2], **layer_kwargs)
        #self.layer3 = self.BigLittleModule(block, drop_block=db_3, *layer_args[2], **layer_kwargs)
        self.layer4 = self._make_layer(block, drop_block=db_4, *layer_args[3], **layer_kwargs)

        # Head (Pooling and Classifier)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_features = 512 * block.expansion
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, reduce_first=1,
                    avg_down=False, down_kernel_size=1, **kwargs):
        downsample = None
        first_dilation = 1 if dilation in (1, 2) else 2
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_args = dict(
                in_channels=self.inplanes, out_channels=planes * block.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=first_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**downsample_args) if avg_down else downsample_conv(**downsample_args)

        block_kwargs = dict(
            cardinality=self.cardinality, base_width=self.base_width, reduce_first=reduce_first,
            dilation=dilation, **kwargs)
        layers = [block(self.inplanes, planes, stride, downsample, first_dilation=first_dilation, **block_kwargs)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes, **block_kwargs) for _ in range(1, blocks)]

        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        del self.fc
        self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes) if num_classes else None

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        #x = self.maxpool(x)

        bx = self.b_conv0(x)
        bx = self.bn_b0(bx)
        lx = self.l_conv0(x)
        lx = self.bn_l0(lx)
        lx = self.act1(lx)
        lx = self.l_conv1(lx)
        lx = self.bn_l1(lx)
        lx = self.act1(lx)
        lx = self.l_conv2(lx)
        lx = self.bn_l2(lx)
        x = self.act1(bx + lx)
        x = self.bl_init(x)
        x = self.bn_bl_init(x)
        x = self.act1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


def blresnet_model(depth, alpha, beta, num_classes=1000, pretrained=False):
    layers = {
        50: [3, 4, 6, 3],
        101: [4, 8, 18, 3],
        152: [5, 12, 30, 3]
    }[depth]
    model = BigLittleResNet(Bottleneck, layers, alpha, beta, num_classes)
    ''' pretrained weights : TODO
    if pretrained:
        url = model_urls['blresnet-{}-a{}-b{}'.format(depth, alpha, beta)]
        checkpoint = torch.load(url)
        model.load_state_dict(checkpoint['state_dict'])
        '''
    return model

