#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2020 
# Apache 2.0

import torch
import torch.nn as nn
import math
from .L2 import L2


class InfoNCELoss(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--l2-reg', type=float, default=0.0001)
        L2.add_args(parser) 
         
    @classmethod
    def build_objective(cls, conf):
        return InfoNCELoss(l2_reg=conf['l2_reg'])

    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration=None):
        return s1 
    
    def __init__(self, avg=True, l2_reg=0.0001):
        super(InfoNCELoss, self).__init__()
        self.avg = avg
        self.l2 = L2()
        self.l2_reg = l2_reg

    def forward(self, model, sample, precomputed=None):
        # Check if we are using precomputed values
        if precomputed is not None:
            x = precomputed
        else:
            x = model(sample)[0]
        
        T = x.size(1) # Length
        B = x.size(0)
        loss = self.compute_loss(x, sample.target)
        # This is for printing. It's the average number of targets classified
        # correctly.
        correct = sum([l.exp() for l in loss]) * T 
        loss = -sum(loss)
        print('InfoNCE: {:0.5f}'.format(math.log(B) - (loss.data.item() / B)), end=' ')
        print('InfoNCE_Acc: {:0.5f}'.format(correct.data.item() / (B * T)), end=' ')
        if self.avg:
            loss /= (B * T)
        
        if self.l2_reg > 0:
            loss_l2, _ = self.l2(model, sample, precomputed=x)
            loss_l2 *= self.l2_reg
            print('L2: {:0.5f}'.format(loss_l2.data.item()), end=' ')
            loss += loss_l2
        return loss, correct

    def compute_loss(self, input, targets):
        input = input.clamp(-30, 30)
        B = input.size(0)
        loss = []
        numerators = input.gather(2, targets.unsqueeze(2)).sum(dim=1)
        for i in range(B):
            denominator = input.gather(
                2, targets[i].repeat(B, 1).unsqueeze(2)
            ).sum(dim=1).logsumexp(dim=0)
            loss.append(numerators[i] - denominator)
        return loss
