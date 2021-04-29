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
        parser.add_argument('--infonce-R', type=int, default=0)
        parser.add_argument('--infonce-momentum', type=float, default=0.9)
        L2.add_args(parser) 
         
    @classmethod
    def build_objective(cls, conf):
        batchsize = eval(conf['datasets'])[0]['batchsize']
        return InfoNCELoss(
            l2_reg=conf['l2_reg'],
            R=conf['infonce_R'],
            K=batchsize,
            momentum=conf['infonce_momentum'],
        )

    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration=None):
        s_ = s1
        s_['B_'] = s1['B_'].logaddexp(math.log(fraction) + s2['B_']) 
        return s_ 
    
    def __init__(self, avg=True, l2_reg=0.0001, R=0, K=32, momentum=0.9):
        super(InfoNCELoss, self).__init__()
        self.avg = avg
        self.l2 = L2()
        self.l2_reg = l2_reg
        self.A = float('-inf') * torch.ones(K)
        self.register_buffer('A_neg', float('-inf') * torch.ones(K)) 
        self.register_buffer('B_', float('-inf') * torch.ones(R+1))
        self.register_buffer('B', float('-inf') * torch.ones(R+1))
        self.R = R
        self.batch_num = 0 # From 0 to R
        self.Y = None
        self.momentum = momentum

    def forward(self, model, sample, precomputed=None):
        # Check if we are using precomputed values
        unsup = sample.target[0, 0] == -1
        if precomputed is not None:
            x = precomputed
        else:
            x = model(sample)[0]
       
        T = x.size(1) # Length
        B = x.size(0)
        # Store the labeled batch
        if not unsup:
            self.Y = sample.target
       
        # Update the smoothing estimate: 
        if self.batch_num == 0:
            self.B.copy_(self.B_)
            self.A_neg.copy_(float('-inf')*torch.ones(1))
        loss = self.compute_loss(x, (not unsup))
        
        # Update next round's smoothing estimate
        if self.batch_num < self.R:
            update = self.A_neg.logsumexp(dim=0) - math.log(B)
            # Missing one negative example (use the average of the others)
            if not unsup:
                update += math.log(1 + 1.0/B)
            self.B_[self.R - self.batch_num - 1] = (
                (
                    math.log(self.momentum)
                    + self.B_[self.R - self.batch_num - 1]
                ).logaddexp(math.log(1. - self.momentum) + update)
            )
 
        # This is for printing. It's the average number of targets classified
        # correctly.
        correct = None
        if not unsup:
            correct = sum([l.exp() for l in loss]) * T 
        loss = -sum(loss)
        self.batch_num = (self.batch_num + 1) % (self.R + 1)
        if not unsup:
            print('InfoNCE: {:0.5f}'.format(math.log((self.R+1)*B) - (loss.data.item() / B)), end=' ')
            print('InfoNCE_Acc: {:0.5f}'.format(correct.data.item() / (B * T)), end=' ')
        if self.avg:
            loss /= (B * T)
        
        if self.l2_reg > 0:
            loss_l2, _ = self.l2(model, sample, precomputed=x)
            loss_l2 *= self.l2_reg
            print('L2: {:0.5f}'.format(loss_l2.data.item()), end=' ')
            loss += loss_l2
        
        return loss, correct

    def compute_loss(self, input, is_sup):
        input = input.clamp(-30, 30)
        B = input.size(0)
        loss = []
        b = self.batch_num
        
        if is_sup:
            numerator = input.gather(2, self.Y.unsqueeze(2)).sum(dim=1)
            self.A = numerator.detach()
        denominators = []
        for i in range(B):
            if is_sup:
                neg_egs = torch.tensor(list(range(0,i)) + list(range(i+1, B))).to(input.device) 
                # Compute the normal InfoNCE denominator
                denominator_neg = input.gather(
                    2, self.Y[i].repeat(B, 1).unsqueeze(2)
                ).sum(dim=1).index_select(0, neg_egs).logsumexp(dim=0)
                denominator = numerator[i].logaddexp(denominator_neg)
            else:
                denominator_neg = input.gather(
                    2, self.Y[i].repeat(B, 1).unsqueeze(2)
                ).sum(dim=1).logsumexp(dim=0)
                denominator = self.A[i].logaddexp(denominator_neg)
                
            # Add in the estimated score of unseen negative examples as well as
            # previously seen suprvised examples (in the case of unsup batch)
            prev_neg = self.A_neg[i].clone()
            denominator_ = denominator.logaddexp(prev_neg).logaddexp(self.B[b])
            denominators.append('{:0.3f}'.format(denominator_.data.item()))
            if is_sup:
                loss.append(numerator[i] - denominator_)
            else:
                loss.append(-denominator_)
            
            self.A_neg[i] = self.A_neg[i].logaddexp(denominator_neg.detach())        
        print("Denominators: ", denominators, end=' ')
        return loss
