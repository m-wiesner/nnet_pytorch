# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified in 2020 for ASR experiments

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, norm=None, leak=.2):
        super(wide_basic, self).__init__()
        self.lrelu = nn.LeakyReLU(leak)
        self.bn1 = get_norm(in_planes, norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = Identity() if dropout_rate == 0.0 else nn.Dropout(p=dropout_rate)
        self.bn2 = get_norm(planes, norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(self.lrelu(self.bn1(x))))
        out = self.conv2(self.lrelu(self.bn2(out)))
        out += self.shortcut(x)

        return out


def get_norm(n_filters, norm):
    if norm is None:
        return Identity()
    elif norm == "batch":
        return nn.BatchNorm2d(n_filters, momentum=0.9)
    elif norm == "instance":
        return nn.InstanceNorm2d(n_filters, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, n_filters)


class SpeechResnet(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--depth', type=int, default=28)
        parser.add_argument('--width', type=int, default=10)
        parser.add_argument('--strides', type=str, default="[1, 1, 2, 2]")

    @classmethod
    def build_model(cls, conf):
        strides = eval(conf.get('strides', "[1, 1, 2, 2]"))
        model = SpeechResnet(
            conf['depth'], conf['width'],
            num_classes=conf['num_targets'],
            strides=strides,
            input_channels=1,
        )
        return model

    def __init__(self, depth, widen_factor, num_classes=10, input_channels=3,
                 sum_pool=False, norm=None, leak=.2, dropout_rate=0.0,
                 strides=[1, 1, 2, 2,]):
        super(SpeechResnet, self).__init__()
        self.leak = leak
        self.odim = num_classes
        self.in_planes = 16
        self.sum_pool = sum_pool
        self.norm = norm
        self.lrelu = nn.LeakyReLU(leak)

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(input_channels, nStages[0], stride=strides[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=strides[1])
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=strides[2])
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=strides[3])
        self.bn1 = get_norm(nStages[3], self.norm)
        self.last_dim = nStages[3]
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, norm=self.norm))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, sample):
        T = sample.input.size(1)
        x = sample.input.unsqueeze(1)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.lrelu(self.bn1(out))
        
        # Get output for different chunk-widths
        cw = sample.input.size(1) - (sample.metadata['left_context'] + sample.metadata['right_context'])
        cw_subsampled = len(range(0, cw, 4))
        output1 = F.adaptive_avg_pool2d(out, (cw_subsampled, 1))
        output0 = self.linear(output1.squeeze(-1).transpose(1, 2))
        return (output0, output1)


class ChainSpeechResnet(SpeechResnet):
    @classmethod
    def build_model(cls, conf):
        strides = eval(conf.get('strides', "[1, 1, 2, 2]"))
        model = ChainSpeechResnet(
            conf['depth'], conf['width'],
            num_classes=conf['num_targets'],
            input_channels=1,
            strides=strides,
        )
        return model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefinal_xent = nn.Linear(
            self.last_dim, 192,
        )
        self.xent_norm = nn.LayerNorm(192)
        self.xent_layer = nn.Linear(192, self.odim)
    
    def forward(self, sample):
        output, x = super().forward(sample)
        if self.training:
            x = self.lrelu(self.prefinal_xent(x.squeeze(-1).transpose(1, 2)))
            x = self.xent_norm(x)
            x = self.xent_layer(x)

        return (output, x)

