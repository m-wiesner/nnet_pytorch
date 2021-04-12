#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2020 
# Apache 2.0

import torch
import torch.nn as nn
import math
from collections import namedtuple
from .L2 import L2


Minibatch = namedtuple('Minibatch', ['input', 'target', 'metadata'])


class InfoNCELoss(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--l2-reg', type=float, default=0.0001)
        parser.add_argument('--small-batchsize', type=int, default=32)
        L2.add_args(parser) 
         
    @classmethod
    def build_objective(cls, conf):
        batchsize = eval(conf['datasets'])[0]['batchsize']
        return InfoNCELoss(
            l2_reg=conf['l2_reg'],
            batchsize=batchsize,
            small_batchsize=conf['small_batchsize'],
        )

    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration=None):
        return s1 
    
    def __init__(self, avg=True, l2_reg=0.0001, batchsize=214, small_batchsize=32):
        super(InfoNCELoss, self).__init__()
        num_small_batches = batchsize // small_batchsize + (batchsize % small_batchsize != 0)
        self.avg = avg
        self.l2 = L2()
        self.l2_reg = l2_reg
        self.small_batchsize = small_batchsize
        self.register_buffer('denom', float('-inf') * torch.ones(batchsize, num_small_batches))

    def first_pass(self, model, sample):
        with torch.no_grad():
            x = model(sample)
        if self.denom.size(1) == 1:
            return 
        B = x[0].size(0)
        # Process each small minibatch
        for i in range(B):
            denominator = x[0].gather(
                2, sample.target[i].repeat(B, 1).unsqueeze(2)
            ).sum(dim=1)
            for b in range(self.denom.size(1) - 1):
                start = b * self.small_batchsize
                end = (b+1) * self.small_batchsize
                neg_egs = torch.tensor(list(range(0, start)) + list(range(end, B))).to(x[0].device) 
                self.denom[i, b] = denominator.index_select(0, neg_egs).logsumexp(dim=0).detach()
            start = (self.denom.size(1) - 1) * self.small_batchsize
            neg_egs = torch.tensor(list(range(0, start))).to(x[0].device) 
            self.denom[i, -1] = denominator.index_select(0, neg_egs).logsumexp(dim=0).detach()

    def split_sample(self, sample):
        B = self.denom.size(1)
        for b in range(B - 1):
            input_tensor = sample.input[b*self.small_batchsize : (b+1)*self.small_batchsize]
            output_tensor = sample.target[b*self.small_batchsize : (b+1)*self.small_batchsize]
            yield Minibatch(input_tensor, output_tensor, sample.metadata)
        input_tensor = sample.input[(B-1)*self.small_batchsize:]
        output_tensor = sample.target[(B-1)*self.small_batchsize:]
        yield Minibatch(input_tensor, output_tensor, sample.metadata)

    def forward(self, model, sample, precomputed=None):
        self.first_pass(model, sample)
        B = sample.input.size(0)
        losses = []
        l2_losses = []
        for b, sample_b in enumerate(self.split_sample(sample)):
            x = model(sample_b)
            loss = self.compute_loss(x[0], sample_b.target, b) 
            losses.extend([l.detach() for l in loss])
            loss_ = -sum(loss) / (B * x[0].size(1))
             
            # We want to compute the loss over the large batch so we cancel out the
            # division by the small batch size by multiplying by it and then
            # dividing by the large batchsize 
            if self.l2_reg > 0:
                loss_l2, _ = self.l2(model, sample, precomputed=x[0])
                loss_l2 *= self.l2_reg * x[0].size(0) / B
                l2_losses.append(loss_l2.detach())
                loss_ += loss_l2 

            if b < self.denom.size(1) - 1:
                loss_.backward()
                loss_.detach()
                del sample_b
                print("Norm_{}: ".format(b), self.compute_norm(model), end=' ')

       
        print("L2: ", l2_losses)
        correct = sum([l.exp() for l in losses]) / B     
        print('InfoNCE_Acc: {:0.5f}'.format(correct.data.item()), end=' ')
        print('InfoNCE: {:0.5f}'.format(math.log(B) + (sum(losses).data.item()/B)), end=' ') 
        return loss_, None

    def compute_loss(self, input, targets, b):
        input = input.clamp(-30, 30)
        B = input.size(0)
        loss = []
        numerators = input.gather(2, targets.unsqueeze(2)).sum(dim=1)
        start = b*self.small_batchsize
        for i in range(B):
            denominator = input.gather(
                2, targets[i].repeat(B, 1).unsqueeze(2)
            ).sum(dim=1).logsumexp(dim=0)
            denominator_ = denominator.logaddexp(self.denom[start+i, b])
            loss.append(numerators[i] - denominator_)
        return loss

    
    def compute_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
