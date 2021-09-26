# Copyright 2021
# Apache 2.0

import torch
import torch.nn.functional as F
from collections import namedtuple
import numpy as np


class MultiBLSTM(torch.nn.Module):
    '''
        Bidirectional LSTM model
    '''
    @staticmethod
    def add_args(parser):
        parser.add_argument('--blstm-idim', type=int, default=64)
        parser.add_argument('--blstm-hdim', type=int, default=512)
        parser.add_argument('--blstm-num-layers', type=int, default=4)
        parser.add_argument('--blstm-dropout', type=float, default=0.1)
        parser.add_argument('--blstm-prefinal-dim', type=int, default=256)
        parser.add_argument('--blstm-num-targets2', type=int, default=2500)
         
    @classmethod
    def build_model(cls, conf):
        model = BLSTM(
            conf['blstm_idim'], conf['num_targets'], conf['blstm_num_targets2'],
            odims=[conf['blstm_hdim'] for i in range(conf['blstm_num_layers'])],
            dropout=conf['blstm_dropout'],
            prefinal_affine_dim=conf['blstm_prefinal_dim'],
            subsample=conf['subsample'],
        )   
        return model
    
    def __init__(
        self, idim, odim1, odim2,
        odims=[512, 512, 512, 512, 512, 512],
        prefinal_affine_dim=512,
        nonlin=F.relu, dropout=0.1, subsample=1
    ):
        super().__init__()
        
        # Proper BLSTM layers
        self.dropout = dropout
        self.nonlin = nonlin
        self.subsample = subsample
        self.blstm = torch.nn.ModuleList()
        self.norm = torch.nn.ModuleList()
        
        next_input_dim = idim
        for cur_odim in odims:
            self.blstm.append(
                torch.nn.LSTM(
                    next_input_dim, cur_odim//2, 1,
                    batch_first=True, bidirectional=True
                )
            )
            self.norm.append(
                torch.nn.BatchNorm1d(cur_odim, eps=1e-03, affine=False)
            )
            next_input_dim = cur_odim

        # Last few layers
        self.prefinal_affine = torch.nn.Linear(
            next_input_dim, prefinal_affine_dim,
        )
        self.norm.append(
            torch.nn.BatchNorm1d(prefinal_affine_dim, eps=1e-03, affine=False)
        )
        self.final_affine1 = torch.nn.Linear(
            prefinal_affine_dim, odim1,
        )
        self.final_affine2 = torch.nn.Linear(
            prefinal_affine_dim, odim2,
        )

    def forward(self, sample):
        xs_pad = sample.input
        left_context = sample.metadata['left_context']
        right_context = sample.metadata['right_context']
        
        # Basic pattern is (blstm, relu, batchnorm, dropout) x num_layers 
        for blstm,norm in zip(self.blstm, self.norm[:-1]):
            xs_pad = blstm(xs_pad)[0]
            xs_pad = self.nonlin(xs_pad)
            # print (xs_pad.shape)
            xs_pad = norm(xs_pad.transpose(1,2)).transpose(1,2)
            xs_pad = F.dropout(xs_pad, p=self.dropout, training=self.training)
      
        # A few final layers
        end_idx = xs_pad.size(1) if right_context == 0 else -right_context
        output2 = xs_pad[:, left_context:end_idx:self.subsample, :]
        xs_pad = self.nonlin(self.prefinal_affine(xs_pad))
        xs_pad = self.norm[-1](xs_pad.transpose(1,2)).transpose(1,2)
        
        # This is basically just glue
        out1 = self.final_affine1(xs_pad)
        out2 = self.final_affine2(xs_pad)
        return (
            out1[:, left_context:end_idx:self.subsample, :],
            out2[:, left_context:end_idx:self.subsample, :],
            output2,
        )


class MultiChainBLSTM(MultiBLSTM):
    @classmethod
    def build_model(cls, conf):
        model = MultiChainBLSTM(
            conf['blstm_idim'], conf['num_targets'], conf['blstm_num_targets2'],
            odims=[conf['blstm_hdim'] for i in range(conf['blstm_num_layers'])],
            dropout=conf['blstm_dropout'],
            prefinal_affine_dim=conf['blstm_prefinal_dim'],
            subsample=conf['subsample'],

        )   
        return model

    def __init__(
        self, idim, odim1, odim2,
        odims=[512, 512, 512, 512, 512, 512],
        prefinal_affine_dim=512,
        nonlin=F.relu, dropout=0.1, subsample=1
    ):
        super().__init__(
            idim, odim1, odim2, odims, prefinal_affine_dim,
            nonlin, dropout, subsample
        )
        self.prefinal_xent = torch.nn.Linear(
            odims[-1],
            prefinal_affine_dim,
        )
        self.xent_norm = torch.nn.BatchNorm1d(
            prefinal_affine_dim, eps=1e-03, affine=False
        )
        self.xent_layer1 = torch.nn.Linear(prefinal_affine_dim, odim1)
        self.xent_layer2 = torch.nn.Linear(prefinal_affine_dim, odim2)
    
    def forward(self, xs_pad):
        out1, out2, xs_pad = super().forward(xs_pad)
        if self.training:
            xs_pad = self.nonlin(self.prefinal_xent(xs_pad))
            xs_pad = self.xent_norm(xs_pad.transpose(1,2)).transpose(1,2)
            xs_pad1 = self.xent_layer1(xs_pad)
            xs_pad2 = self.xent_layer2(xs_pad)
        else:
            xs_pad1 = xs_pad
            xs_pad2 = xs_pad
        return out1, xs_pad1, out2, xs_pad2 

