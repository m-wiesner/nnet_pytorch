import torch
import torch.nn.functional as F
from collections import namedtuple
import numpy as np


class BLSTM(torch.nn.Module):
    '''
        Bidirectional LSTM model
    '''
    @staticmethod
    def add_args(parser):
        parser.add_argument('--blstm-hdim', type=int, default=512)
        parser.add_argument('--blstm-num-layers', type=int, default=4)
        parser.add_argument('--blstm-dropout', type=float, default=0.1)
        parser.add_argument('--blstm-prefinal-dim', type=int, default=256)
         
    @classmethod
    def build_model(cls, conf):
        model = BLSTM(
            conf['idim'], conf['num_targets'],
            odims=[conf['blstm_hdim'] for i in range(conf['blstm_num_layers'])],
            dropout=conf['blstm_dropout'],
            prefinal_affine_dim=conf['blstm_prefinal_dim'],
            subsample=conf['subsample'],
            batch_norm_dropout=True
        )   
        return model
    
    def __init__(
        self, idim, odim,
        odims=[512, 512, 512, 512, 512, 512],
        prefinal_affine_dim=512,
        nonlin=F.relu, dropout=0.1, subsample=1, batch_norm_dropout=True
    ):
        super().__init__()
        
        # Proper BLSTM layers
        self.batch_norm_dropout = batch_norm_dropout
        self.dropout = dropout
        self.nonlin = nonlin
        self.subsample = subsample
        self.blstm = torch.nn.ModuleList()
        self.batchnorm = torch.nn.ModuleList()
        
        next_input_dim = idim
        for cur_odim in odims:
            self.blstm.append(
                torch.nn.LSTM(
                    next_input_dim, cur_odim//2, 1,
                    batch_first=True, bidirectional=True
                )
            )
            self.batchnorm.append(
                torch.nn.BatchNorm1d(cur_odim, eps=1e-03, affine=False)
            )
            next_input_dim = cur_odim

        # Last few layers
        self.prefinal_affine = torch.nn.Linear(
            next_input_dim, prefinal_affine_dim,
        )
        self.batchnorm.append(
            torch.nn.BatchNorm1d(
                prefinal_affine_dim, eps=1e-03, affine=False
            )
        )
        self.final_affine = torch.nn.Linear(
            prefinal_affine_dim, odim,
        )

    def forward(self, sample):
        xs_pad = sample.input
        left_context = sample.metadata['left_context']
        right_context = sample.metadata['right_context']
       
        # Basic pattern is (blstm, relu, batchnorm, dropout) x num_layers 
        for blstm, batchnorm in zip(self.blstm, self.batchnorm[:-1]):
            xs_pad = blstm(xs_pad)[0].transpose(0,1)
            xs_pad = self.nonlin(xs_pad)
            if self.batch_norm_dropout: 
                xs_pad = batchnorm(xs_pad)
                xs_pad = F.dropout(xs_pad, p=self.dropout, training=self.training)
      
        # A few final layers
        end_idx = xs_pad.size(1) if right_context == 0 else -right_context
        output2 = xs_pad[:, left_context:end_idx:self.subsample, :]
        xs_pad = self.nonlin(self.prefinal_affine(xs_pad))
        if self.batch_norm_dropout:
            xs_pad = self.batchnorm[-1](xs_pad)
        
        # This is basically just glue
        output = self.final_affine(xs_pad)
        return (
            output[:, left_context:end_idx:self.subsample, :],
            output2,
        )


class ChainBLSTM(BLSTM):
    @classmethod
    def build_model(cls, conf):
        model = ChainBLSTM(
            conf['idim'], conf['num_targets'],
            odims=[conf['blstm_hdim'] for i in range(conf['blstm_num_layers'])],
            dropout=conf['blstm_dropout'],
            prefinal_affine_dim=conf['blstm_prefinal_dim'],
            subsample=conf['subsample'],
            batch_norm_dropout=True
        )   
        return model

    def __init__(
        self, idim, odim,
        odims=[512, 512, 512, 512, 512, 512],
        prefinal_affine_dim=512,
        nonlin=F.relu, dropout=0.1, subsample=1, batch_norm_dropout=True
    ):
        super().__init__(
            idim, odim, odims, prefinal_affine_dim,
            nonlin, dropout, subsample
        )
        self.prefinal_xent = torch.nn.Linear(
            odims[-1],
            prefinal_affine_dim,
        )
        self.xent_batchnorm = torch.nn.BatchNorm1d(
            prefinal_affine_dim,
            eps=1e-03, affine=False
        )
        self.xent_layer = torch.nn.Linear(prefinal_affine_dim, odim)
    
    def forward(self, xs_pad):
        output, xs_pad = super().forward(xs_pad)
        if self.training:
            xs_pad = self.nonlin(self.prefinal_xent(xs_pad))
            if self.batch_norm_dropout:
                xs_pad = self.xent_batchnorm(xs_pad)
            xs_pad = self.xent_layer(xs_pad)
        return output, xs_pad 

