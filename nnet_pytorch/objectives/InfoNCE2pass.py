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


class InfoNCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, targets, loss, grad):
        ctx.save_for_backward(grad) 
        return sum(loss) 
    
    @staticmethod
    def backward(ctx, objf_grad):
        num_grad, = ctx.saved_tensors
        num_grad = torch.mul(num_grad, objf_grad) 
        return num_grad, None, None, None
        

class InfoNCELoss(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--l2-reg', type=float, default=0.0001)
        parser.add_argument('--small-batchsize', type=int, default=32)
        parser.add_argument('--sup-batchsize', type=int, default=32)
        L2.add_args(parser) 
         
    @classmethod
    def build_objective(cls, conf):
        batchsize = eval(conf['datasets'])[0]['batchsize']
        sup_batchsize = conf.get('sup_batchsize', batchsize)

        # Maximum supervised batchsize
        sup_batchsize = min(batchsize, sup_batchsize)

        return InfoNCELoss(
            l2_reg=conf['l2_reg'],
            batchsize=batchsize,
            small_batchsize=conf['small_batchsize'],
            sup_batchsize=sup_batchsize,
        )

    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration=None):
        return s1 
    
    def __init__(self, avg=True, l2_reg=0.0001, batchsize=214, small_batchsize=32, sup_batchsize=214):
        super(InfoNCELoss, self).__init__()
        num_small_batches = batchsize // small_batchsize + (batchsize % small_batchsize != 0)
        self.avg = avg
        self.l2 = L2()
        self.l2_reg = l2_reg
        self.small_batchsize = small_batchsize
        self.num_small_batches = num_small_batches
        self.sup_batchsize = sup_batchsize

    def loss_fun(self, input, targets):
        with torch.no_grad():
            input = input.clamp(-30, 30)
            B = targets.size(0)
            loss = []
            numerators = input.gather(2, targets.unsqueeze(2)).sum(dim=1)
            grad = torch.zeros_like(input)
            for i in range(self.sup_batchsize):
                batch_idx = torch.LongTensor([i]).to(input.device) 
                terms = input.gather(
                    2, targets[i].repeat(B, 1).unsqueeze(2)
                ).sum(dim=1)
                denominator = terms.logsumexp(dim=0)
                grad_mask = torch.zeros_like(grad)
                grad_mask.scatter_(2, targets[i].repeat(B, 1).unsqueeze(2), 1.0)
                batch_mask = torch.zeros_like(terms)
                grad += grad_mask * (batch_mask.scatter_(0, batch_idx.unsqueeze(-1), 1.0) - (terms - denominator).exp()).unsqueeze(-1)  
                loss.append(numerators[i] - denominator) 
        return loss, grad 

    def split_sample(self, sample):
        B = self.num_small_batches
        for b in range(B - 1):
            input_tensor = sample.input[b*self.small_batchsize : (b+1)*self.small_batchsize]
            output_tensor = sample.target[b*self.small_batchsize : (b+1)*self.small_batchsize]
            yield Minibatch(input_tensor, output_tensor, sample.metadata)
        input_tensor = sample.input[(B-1)*self.small_batchsize:]
        output_tensor = sample.target[(B-1)*self.small_batchsize:]
        yield Minibatch(input_tensor, output_tensor, sample.metadata)

    def forward(self, model, sample, precomputed=None):
        with torch.no_grad():
            x = model(sample)
        losses, grad = self.loss_fun(x[0], sample.target)

        B = sample.input.size(0)
        l2_losses = []
        for b, sample_b in enumerate(self.split_sample(sample)):
            x = model(sample_b)
            start = b*self.small_batchsize # Which targets to use
            end = (b+1) * self.small_batchsize
            if start < len(losses): 
                losses_b = losses[start:end]
            else:
                losses_b = losses
            loss = InfoNCEFunction.apply(
                x[0], sample_b.target,
                losses_b, grad[start:end],
            ) 
            loss_ = -loss / (self.sup_batchsize * x[0].size(1))
             
            # We want to compute the loss over the large batch so we cancel out the
            # division by the small batch size by multiplying by it and then
            # dividing by the large batchsize 
            if self.l2_reg > 0:
                loss_l2, _ = self.l2(model, sample, precomputed=x[0])
                loss_l2 *= self.l2_reg * x[0].size(0) / B
                l2_losses.append(loss_l2.detach())
                loss_ += loss_l2 

            if b < self.num_small_batches - 1:
                loss_.backward()
                loss_.detach()
                del sample_b
                #print("Norm_{}: ".format(b), self.compute_norm(model), end=' ')

       
        print("L2: ", sum(l2_losses).item(), end=' ')
        correct = sum([l.exp() for l in losses]) / self.sup_batchsize     
        print('InfoNCE_Acc: {:0.5f}'.format(correct.data.item()), end=' ')
        print('InfoNCE: {:0.5f}'.format(math.log(B) + (sum(losses).data.item()/self.sup_batchsize)), end=' ') 
        return loss_, None

    def compute_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
