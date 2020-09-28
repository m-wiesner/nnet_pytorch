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
    def forward(ctx, input, targets):
        input = input.clamp(-30, 30).exp()
        B = input.size(0)
        num_grad = torch.zeros_like(input)
        num_grad.scatter_(2, targets.unsqueeze(2), 1.0) 
        ctx.save_for_backward(num_grad)
        acoustic_cost = sum(
            [
                input[i, t, targets[i, t]]
                    for i in range(input.size(0))
                        for t in range(input.size(1))
            ]
        )
        return acoustic_cost

    @staticmethod
    def backward(ctx, objf_grad):
        num_grad, = ctx.saved_tensors
        num_grad = torch.mul(num_grad, objf_grad)
        return num_grad, None


class ChainLoss(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--denom-graph', required=True)
    
    @classmethod
    def build_objective(cls, conf):
        return ChainLoss(
            conf['denom_graph'],
            conf['num_targets'],
        )

    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration=None):
        return s1 
    
    def __init__(self, den_graph, avg=True):
        super(ChainLoss, self).__init__()
        self.den_graph = ChainGraph(
            fst=simplefst.StdVectorFst.read(den_graph),
        )
        self.avg = avg

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
        den_objf = ChainFunction.apply(x, x_lengths, den_graphs, 0.1)   
        num_objf = NumeratorFunction.apply(x, sample.target)
        loss = -(num_objf - den_objf)
        if self.avg:
            loss /= (B * T)
        return loss, None
