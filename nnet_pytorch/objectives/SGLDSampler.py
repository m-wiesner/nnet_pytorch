#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2020
# Apache 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import numpy as np
import random
import sys
from .AcceleratedSGLD import AcceleratedSGLD
from .SGLD import SGLD  


Samples = namedtuple('Samples', ['input', 'metadata']) 


class SGLDSampler(object):
    def __init__(self,
        buffer_size = 10000,
        sgld_reinit_p=0.05,
        sgld_stepsize=1.0,
        sgld_noise=1.0,
        num_steps=20,
        sgld_thresh=0.001,
        sgld_optim='accsgld',
        sgld_replay_correction=1.0,
        sgld_weight_decay=1e-05,
    ):
        self.buffer = torch.FloatTensor()
        self.buffer_numsteps = torch.zeros(buffer_size) 
        self.reinit_p = sgld_reinit_p
        self.stepsize = sgld_stepsize
        self.noise = sgld_noise
        self.num_steps = num_steps
        self.buffersize = buffer_size
        self.sgld_thresh = sgld_thresh
        self.optim = sgld_optim
        self.replay_correction=sgld_replay_correction
        self.weight_decay=sgld_weight_decay
    
    def init_random(self, bs, cw, dim, std):
        x = torch.FloatTensor(bs, cw, dim).uniform_(-1, 1)
        return x

    def sample_like(self, sample):
        '''
            Assumes inputs are mean normalized
        '''
        x = sample.input
        metadata = sample.metadata
        std = x.flatten(0, 1).std(dim=0).cpu()
        # The first minibatch
        if len(self.buffer) == 0:
            self.buffer = self.init_random(self.buffersize, x.size(1), x.size(2), std)
        # Sample batchsize -- x is B x T x D -- number of random indices
        idxs = torch.randint(0, len(self.buffer), (x.size(0),)) 
        # Get random samples from buffer
        buffer_samples = self.buffer[idxs]
        buffer_numsteps = self.buffer_numsteps[idxs] 
        # Generate batchsize number of random samples
        random_samples = self.init_random(x.size(0), x.size(1), x.size(2), std)
        random_numsteps = torch.zeros(x.size(0))
        choose_random = (torch.rand(x.size(0)) < self.reinit_p).float()[:, None, None]
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples 
        numsteps = choose_random.view(-1) * random_numsteps + (1 - choose_random.view(-1)) * buffer_numsteps 
        return samples.to(x.device), idxs, metadata, numsteps.to(x.device)

    def update(self, x, f, sample_energy=None, max_iters=150):
        # Debug Diagnostic
        x_k = torch.autograd.Variable(x[0], requires_grad=True)
        optimizers = {
            'sgd': SGLD(
                [x_k], lr=self.stepsize, momentum=0.0, noise=self.noise,
                stepscale=self.replay_correction, clamp=0.0, nesterov=False,
                weight_decay=self.weight_decay,
            )
        }
        if sample_energy is not None:
            optimizers = {
                    **optimizers,
                    'accsgld': AcceleratedSGLD(
                    [x_k], sample_energy, lr=self.stepsize, noise=self.noise,
                    stepscale=self.replay_correction, clamp=1.0,
                    weight_decay=self.weight_decay,
                ),
            }
        else:
            print('ERROR: Sample Energy must be passed as argument in this'
             ' version. Older versions using other optimizers are no longer'
             ' used.', file=sys.stderr
            )
            sys.exit(1)
        
        optim = optimizers[self.optim] 
        print('-------------------- Sampling -----------------------------')
        y = f(Samples(x_k, x[2]))
        not_converged = True
        numsteps = x[3]
        print('k: ', 0, ' --- E: ', y.data.item(), end=' --- ')
        print('steps: ', numsteps.mean().item(), ' --- num-new: ', (numsteps == 0).sum().item(), end=' --- ')
        k = 0
        while not_converged:
            x_k.grad = torch.autograd.grad(y, [x_k], retain_graph=False)[0].clone()
            grad_norm = torch.nn.utils.clip_grad_norm_([x_k], 5.0) 
            print('var: ', x_k.std().data.item(), 'mean: ', x_k.mean().data.item(), 'grad: ', grad_norm, ' --- noise: ', self.noise) 
            numsteps += 1
            if self.optim == 'accsgld': 
                optim.step(numsteps=numsteps, startval=y.data.item()) 
            else:
                optim.step(numsteps=numsteps)
            optim.zero_grad()
            y = f(Samples(x_k, x[2]))
            print('k: ', k+1, ' --- E: ', y.data.item(), end=' --- ')
            k += 1
            # Check for convergence
            if sample_energy is not None:
                not_converged = (((y.data.item() - sample_energy) / abs(sample_energy)) > self.sgld_thresh)
            not_converged = (not_converged or (k < self.num_steps)) and (k < max_iters) 
        print('\n------------------------------------------------------------')
        x_ = x_k.detach()
        if len(self.buffer) > 0:
            self.buffer[x[1]] = x_.cpu()
            self.buffer_numsteps[x[1]] = numsteps.cpu()
        return x_, k
 
