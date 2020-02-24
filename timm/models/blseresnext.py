# -*- coding: utf-8 -*-

# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

#this module consolidates blresnet, blresnext and blseresnext
#when not otherwise specified basewidth and cardinality each default to 64 and 1 which is the default for resnet50
#use_se defaults to False which devolves the model into non-se

#TODO : further adapt this model to follow the style of pytorch image models by rwightman
#TODO : integrate Activation layer factory
#TODO : integrate Norm layer factory
#TODO : integrate Attention layer factory
#TODO : style match parameters(capitalization, order, etc)

#TODO : make all but bLSEResNeXt(and maybe blModule), imported from pytorch image model resnet
#TODO : finally, only leave plain blresnet?? and/or blModule here

import math

import torch
import torch.nn as nn

model_urls = {
    'blresnet-50-a2-b4': 'pretrained/ImageNet-bLResNet-50-a2-b4.pth.tar',
    'blresnet-101-a2-b4': 'pretrained/ImageNet-bLResNet-101-a2-b4.pth.tar',
    'blresnext-50-32x4d-a2-b4': 'pretrained/ImageNet-bLResNeXt-50-32x4d-a2-b4.pth.tar',
    'blresnext-101-32x4d-a2-b4': 'pretrained/ImageNet-bLResNeXt-101-32x4d-a2-b4.pth.tar',
    'blresnext-101-64x4d-a2-b4': 'pretrained/ImageNet-bLResNeXt-101-64x4d-a2-b4.pth.tar',
    'blseresnext-50-32x4d-a2-b4': 'pretrained/ImageNet-bLSEResNeXt-50-32x4d-a2-b4.pth.tar',
    'blseresnext-101-32x4d-a2-b4': 'pretrained/ImageNet-bLSEResNeXt-101-32x4d-a2-b4.pth.tar',
}

__all__ = ['blresnet_model','blseresnext_model','blresnext_model']


class SEModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, basewidth=64, cardinality=1, stride=1, downsample=None, last_relu=True, use_se=True):
        super(Bottleneck, self).__init__()

        #make C=1 default and make that case the same as plain resnet without groups to consolidate blresnet with blseresnext.py
        # D * C is planes // self.expansion in case of plain resnets
        
        C = cardinality
        D = int(math.floor(planes * (basewidth / 64.0))) // self.expansion if C is not 1 else planes // self.expansion

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=C)
                               #when groups = 1 this is just plain old convolution

        self.bn2 = nn.BatchNorm2d(D*C)
        self.conv3 = nn.Conv2d(D*C, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

        self.se_layer = SEModule(planes, 16) if use_se else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se_layer is not None:
             out = self.se_layer(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.last_relu:
            out = self.relu(out)

        return out


class bLModule(nn.Module):
    def __init__(self, block, in_channels, out_channels, blocks, basewidth=64, cardinality=1, alpha=2, beta=4, stride=1):
        super(bLModule, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.big = self._make_layer(block, in_channels, out_channels, blocks - 1,
                                    basewidth, cardinality, 2, last_relu=False)
        self.little = self._make_layer(block, in_channels, out_channels // alpha,
                                       max(1, blocks // beta - 1), basewidth * alpha, cardinality // alpha)
        self.little_e = nn.Sequential(
            nn.Conv2d(out_channels // alpha, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))

        self.fusion = self._make_layer(block, out_channels, out_channels, 1, basewidth, cardinality, stride=stride)

    def _make_layer(self, block, inplanes, planes, blocks, basewidth, cardinality, stride=1, last_relu=True):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, padding=0, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
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
        big = torch.nn.functional.interpolate(big, little.shape[2:])
        out = self.relu(big + little)
        out = self.fusion(out)

        return out


class bLSEResNeXt(nn.Module):

    def __init__(self, block, layers, basewidth=64, cardinality=1, alpha=2, beta=4, use_se=True, num_classes=1000):
        super(bLSEResNeXt, self).__init__()
        num_channels = [64, 128, 256, 512]
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.relu = nn.ReLU(inplace=True)

        self.b_conv0 = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_b0 = nn.BatchNorm2d(num_channels[0])
        self.l_conv0 = nn.Conv2d(num_channels[0], num_channels[0] // alpha,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_l0 = nn.BatchNorm2d(num_channels[0] // alpha)
        self.l_conv1 = nn.Conv2d(num_channels[0] // alpha, num_channels[0] //
                                 alpha, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_l1 = nn.BatchNorm2d(num_channels[0] // alpha)
        self.l_conv2 = nn.Conv2d(num_channels[0] // alpha, num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_l2 = nn.BatchNorm2d(num_channels[0])

        self.bl_init = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_bl_init = nn.BatchNorm2d(num_channels[0])
        self.layer1 = bLModule(block, num_channels[0], num_channels[0] * block.expansion,
                               layers[0], basewidth, cardinality, alpha, beta, stride=2)
        self.layer2 = bLModule(block, num_channels[0] * block.expansion, num_channels[1]
                               * block.expansion, layers[1], basewidth, cardinality, alpha, beta, stride=2)
        self.layer3 = bLModule(block, num_channels[1] * block.expansion, num_channels[2]
                               * block.expansion, layers[2], basewidth, cardinality, alpha, beta, stride=1)
        self.layer4 = self._make_layer(block, num_channels[2] * block.expansion,
                                       num_channels[3] * block.expansion, layers[3], basewidth, cardinality, stride=2)
        self.gappool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each block.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, inplanes, planes, blocks, basewidth, cardinality, stride=1):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, padding=0, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)

        layers = []
        layers.append(block(inplanes, planes, basewidth, cardinality, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, basewidth, cardinality))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        bx = self.b_conv0(x)
        bx = self.bn_b0(bx)

        lx = self.l_conv0(x)
        lx = self.bn_l0(lx)
        lx = self.relu(lx)
        lx = self.l_conv1(lx)
        lx = self.bn_l1(lx)
        lx = self.relu(lx)
        lx = self.l_conv2(lx)
        lx = self.bn_l2(lx)

        x = self.relu(bx + lx)
        x = self.bl_init(x)
        x = self.bn_bl_init(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gappool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def blresnet_model(depth, alpha, beta, num_classes=1000, pretrained=False):
    layers = {
        50: [3, 4, 6, 3],
        101: [4, 8, 18, 3],
        152: [5, 12, 30, 3]
    }[depth]
    model = bLSEResNeXt(Bottleneck, layers, alpha, beta, num_classes)

    if pretrained:
        url = model_urls['blresnet-{}-a{}-b{}'.format(depth, alpha, beta)]
        checkpoint = torch.load(url)
        model.load_state_dict(checkpoint['state_dict'])
    return model

def blresnext_model(depth, basewidth, cardinality, alpha, beta, use_se=False,
                    num_classes=1000, pretrained=False):
    layers = {
        50: [3, 4, 6, 3],
        101: [4, 8, 18, 3],
        152: [5, 12, 30, 3]
    }[depth]

    model = bLSEResNeXt(Bottleneck, layers, basewidth, cardinality,
                      alpha, beta, use_se, num_classes)
    if pretrained:
        url = model_urls['blresnext-{}-{}x{}d-a{}-b{}'.format(depth, cardinality,
                                                              basewidth, alpha, beta)]
        checkpoint = torch.load(url)
        model.load_state_dict(checkpoint['state_dict'])

    return model

def blseresnext_model(depth, basewidth, cardinality, alpha, beta, use_se=True,
                      num_classes=1000, pretrained=False):
    layers = {
        50: [3, 4, 6, 3],
        101: [4, 8, 18, 3],
        152: [5, 12, 30, 3]
    }[depth]

    model = bLSEResNeXt(Bottleneck, layers, basewidth, cardinality,
                        alpha, beta, num_classes,use_se)
    if pretrained:
        url = model_urls['blseresnext-{}-{}x{}d-a{}-b{}'.format(depth, cardinality,
                                                                basewidth, alpha, beta)]
        checkpoint = torch.load(url)
        model.load_state_dict(checkpoint['state_dict'])

    return model
