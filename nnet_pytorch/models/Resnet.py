import torch
import torch.nn.functional as F
from collections import namedtuple
import numpy as np


Layer = namedtuple(
    'Layer',
    ['odim', 'num_blocks', 'dilation']
)

DefaultLayer = Layer(512, 2, 1)


class Bottleneck(torch.nn.Module):
    def __init__(self, idim, odim, bndim, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv1d(
            idim, bndim, stride=1, kernel_size=1, bias=False,
        )
        #self.n1 = torch.nn.GroupNorm(1, bndim)
        self.n1 = torch.nn.Identity()
        self.conv2 = torch.nn.Conv1d(
            bndim, bndim, stride=stride, kernel_size=3,
            dilation=dilation, padding=dilation, bias=False,
        )
        #self.n2 = torch.nn.GroupNorm(1, bndim)
        self.n2 = torch.nn.Identity()
        self.conv3 = torch.nn.Conv1d(
            bndim, odim, stride=1, kernel_size=1, bias=False, 
        )
        #self.n3 = torch.nn.GroupNorm(1, odim)
        self.n3 = torch.nn.Identity()
        self.relu = torch.nn.ReLU(inplace=True)
        self.identity = torch.nn.Sequential(
            torch.nn.Conv1d(
                idim, odim, stride=stride, kernel_size=1, bias=False,
            ),
            #torch.nn.GroupNorm(1, odim),
            torch.nn.Identity()
        )

    def forward(self, x):
        identity = self.identity(x)
        out = self.conv1(x)
        out = self.n1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.n2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.n3(out)
        
        out += identity
        out = self.relu(out)
        
        return out 


class SpeechResnet(torch.nn.Module):
    '''
        Kaldi TDNN style encoder implemented as convolutions
    '''
    @staticmethod
    def add_args(parser):
        parser.add_argument('--resnet-hdim', type=int, default=192)
        parser.add_argument('--resnet-bottleneck', type=int, default=96)
        parser.add_argument('--resnet-layers',
            default="[[256, 3, 1], [512, 3, 1], [512, 1, 3], [512, 3, 1]]"
        )
         
    @classmethod
    def build_model(cls, conf):   
        args_layers = eval(conf['resnet_layers'])
        layers = []
        for l in args_layers:
            layers.append(Layer(*l))
        
        model = SpeechResnet(
            conf['idim'], conf['num_targets'],
            hdim=conf['resnet_hdim'],
            bndim=conf['resnet_bottleneck'],
            layers=layers
        ) 
        return model
    
    def __init__(
        self, idim, odim,
        layers=[DefaultLayer]*4, hdim=64, bndim=64,
    ):
        '''
            layers is a list of tuples, one for each of the 4 layers.
            It has the format (idim, odim, num_blocks, dilation)
        '''
        super(SpeechResnet, self).__init__()
        self.idim = idim
        self.odim = odim
        self.hdim = hdim
        self.bndim = bndim
        self.layers = layers
        
        self.conv1 = torch.nn.Conv1d(
            idim, hdim, kernel_size=3, stride=1, bias=False, padding=1
        ) 
        self.n1 = torch.nn.GroupNorm(1, hdim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(
            hdim, layers[0].odim, layers[0].num_blocks,
            stride=layers[0].dilation, dilation=layers[0].dilation,
        )
        self.layer2 = self._make_layer(
            layers[0].odim, layers[1].odim, layers[1].num_blocks,
            stride=layers[1].dilation, dilation=layers[1].dilation,
        )
        self.layer3 = self._make_layer(
            layers[1].odim, layers[2].odim, layers[2].num_blocks,
            stride=layers[2].dilation, dilation=layers[2].dilation,
        )
        self.layer4 = self._make_layer(
            layers[2].odim, layers[3].odim, layers[3].num_blocks,
            stride=layers[3].dilation, dilation=layers[3].dilation,
        )
        self.prefinal_affine = torch.nn.Linear(layers[3].odim, self.hdim)
        self.n2 = torch.nn.LayerNorm(hdim)
        self.fc = torch.nn.Linear(self.hdim, odim) 
   
        # Initialization 
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.GroupNorm):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, idim, odim, num_blocks, stride=1, dilation=1):
        layers = []
        for i in range(num_blocks):
            if i > 0:
                idim = odim
            layers.append(
                Bottleneck(
                    idim, odim, self.bndim, stride=stride, dilation=dilation, 
                )
            )
        return torch.nn.Sequential(*layers)
         
    def forward(self, sample):
        x = sample.input
        # x is (B, L, D), but for convolutions we expect (B, #C, L)
        # We treat the dimensions as different channels here so we
        # swap the number of channels and the length.
        x = x.transpose(1, 2)

        # Initial processing
        x = self.conv1(x)
        #x = self.n1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Main Resnet layers and output the resnet outputs for potential
        # multitask processing later on
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        output1 = self.layer4(x)

        # Output the chain block
        #output = self.avgpool(output1)
        output0 = self.prefinal_affine(output1.transpose(1, 2))
        #output0 = self.n2(output0)
        output0 = self.relu(output0) 
        output0 = self.fc(output0) 
        return (output0, output1)  


class ChainSpeechResnet(SpeechResnet):
    @classmethod
    def build_model(cls, conf):   
        args_layers = eval(conf['resnet_layers'])
        layers = []
        for l in args_layers:
            layers.append(Layer(*l))
        
        model = ChainSpeechResnet(
            conf['idim'], conf['num_targets'],
            hdim=conf['resnet_hdim'],
            bndim=conf['resnet_bottleneck'],
            layers=layers
        ) 
        return model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefinal_xent = torch.nn.Linear(
            self.layers[-1].odim, self.hdim * 2
        )
        self.xent_norm = torch.nn.LayerNorm(self.hdim * 2)
        self.xent_layer = torch.nn.Linear(self.hdim * 2, self.odim)
    
    def forward(self, sample):
        output, x = super().forward(sample)
        if self.training:
            x = self.relu(self.prefinal_xent(x.transpose(1, 2)))
            x = self.xent_norm(x)
            x = self.xent_layer(x)

        return (output, x)

