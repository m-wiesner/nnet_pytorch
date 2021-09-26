#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2020 
# Apache 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from .pychain.pychain.graph import ChainGraphBatch, ChainGraph
import pychain_C
import simplefst
from .pychain.pychain.chain import ChainFunction 


class NumeratorFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, targets):
        input = input.clamp(-30, 30)
        output = input.gather(2, targets.unsqueeze(2)).sum()
        B = input.size(0)
        num_grad = torch.zeros_like(input)
        num_grad.scatter_(2, targets.unsqueeze(2), 1.0) 
        kernel = torch.FloatTensor([[[0.1, 0.8, 0.1]]]).repeat(input.size(-1), 1, 1).to(input.device)
        num_grad = F.conv1d(
            num_grad.transpose(1, 2),
            kernel, stride=1, groups=input.size(-1),
            padding=1
        ).transpose(1, 2)
        ctx.save_for_backward(num_grad)
        return output 

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, objf_grad):
        num_grad, = ctx.saved_tensors
        num_grad = torch.mul(num_grad, objf_grad)
        return num_grad, None


class ChainLoss(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--denom-graph', required=True)
        parser.add_argument('--leaky-hmm', type=float, default=0.1)
    
    @classmethod
    def build_objective(cls, conf):
        return ChainLoss(
            conf['denom_graph'],
            avg=True, 
            leaky_hmm=conf.get('leaky_hmm', 0.1)
        )

    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration=None):
        return s1 
    
    def __init__(self, den_graph, avg=True, leaky_hmm=0.1):
        super(ChainLoss, self).__init__()
        self.den_graph = ChainGraph(
            fst=simplefst.StdVectorFst.read(den_graph),
        )
        self.avg = avg
        self.leaky_hmm = leaky_hmm

    def forward(self, model, sample, precomputed=None):
        B = sample.input.size(0) # batchsize
        den_graphs = ChainGraphBatch(self.den_graph, B)
        
        # Check if we are using precomputed values
        if precomputed is not None:
            x = precomputed
        else:
            x = model(sample)[0]
        
        T = x.size(1) # Length
        x_lengths = torch.LongTensor([T] * B).to(x.device)
        den_objf = ChainFunction.apply(x, x_lengths, den_graphs, self.leaky_hmm) 
        num_objf = NumeratorFunction.apply(x, sample.target)
        loss = -(num_objf - den_objf)
        if self.avg:
            loss /= (B * T)
        return loss, None
