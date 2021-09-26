# Copyright 2021
# Apache 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from .L2 import L2
from .CrossEntropy import CrossEntropy 
from .LFMMIOnly import ChainLoss as LFMMI
from .InfoNCEOnly import InfoNCELoss as InfoNCE


Minibatch = namedtuple('Minibatch', ['input', 'target', 'metadata'])


class ChainLoss(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--xent-reg', type=float, default=0.2)
        parser.add_argument('--infonce-reg', type=float, default=0.2)
        parser.add_argument('--l2-reg', type=float, default=0.00025)
        for m in [L2, CrossEntropy, LFMMI]:
            m.add_args(parser) 

    @classmethod
    def build_objective(cls, conf):
        return ChainLoss(
            conf['denom_graph'],
            xent_reg=conf['xent_reg'],
            infonce_reg=conf.get('infonce_reg', 0),
            l2_reg=conf['l2_reg'],
            leaky_hmm=conf.get('leaky_hmm', 0.1),
        )

    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration=None):
        return s1 

    def __init__(
        self, den_graph,
        xent_reg=0.2, infonce_reg=0.0, l2_reg=0.00025, avg=True, leaky_hmm=0.1,
    ):
        super(ChainLoss, self).__init__()
        self.lfmmi = LFMMI(den_graph, leaky_hmm=leaky_hmm)  
        self.xent = CrossEntropy()
        self.infonce_reg = infonce_reg
        if infonce_reg > 0:
            self.infonce = InfoNCE(l2_reg=0.0)
        self.l2 = L2()
        
        self.l2_reg = l2_reg
        self.xent_reg = xent_reg

    def forward(self, model, sample, precomputed=None):
        if precomputed is not None:
            chain_output = precomputed
        else:
            chain_output = model(sample)
        
        losses = [] 
        correct = None
        # LFMMI
        loss_lfmmi, _ = self.lfmmi(
            model,
            sample,
            precomputed=chain_output[0],
        )
        print('LFMMI: {:0.5f}'.format(loss_lfmmi.data.item()), end=' ')
        losses.append(loss_lfmmi)
        # XENT
        if self.xent_reg > 0:
            loss_xent, correct = self.xent(
                model,
                sample,
                precomputed=chain_output[1],
            )
            loss_xent *= self.xent_reg
            print('XENT: {:0.5f}'.format(loss_xent.data.item()), end=' ')
            losses.append(loss_xent)
        
        # L2
        if self.l2_reg > 0:
            loss_l2, _ = self.l2(
                model,
                sample,
                precomputed=chain_output[0],
            )
            loss_l2 *= self.l2_reg
            print('L2: {:0.5f}'.format(loss_l2.data.item()), end=' ')
            losses.append(loss_l2)

        # InfoNCE
        if self.infonce_reg > 0:
            B = chain_output[0].size(0)
            T = chain_output[0].size(1)
            infonce_input = chain_output[0].view(B * T, 1, -1) 
            sample2 = Minibatch(sample.input, sample.target.view(-1, 1), sample.metadata) 
            loss, _ = self.infonce(
                model,
                sample2,
                precomputed=infonce_input,
            )
            loss *= self.infonce_reg
            losses.append(loss)

        loss = sum(losses)
        return loss, correct


