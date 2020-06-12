# Copyright  2020

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0

# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .SGLDSampler import SGLDSampler
from functools import partial
from .L2 import L2
from collections import namedtuple
import math


Samples = namedtuple('Samples', ['input', 'metadata']) 


class Energy(nn.Module):
    def __init__(self):
        super(Energy, self).__init__()
    
    def forward(self, model, sample):
        output = model(sample)
        objf = output[0].logsumexp(-1).sum()
        return -objf 


class EBMLoss(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--sgld-steps', type=int, default=20)
        parser.add_argument('--sgld-buffer', type=int, default=10000)
        parser.add_argument('--sgld-reinit-p', type=float, default=0.05)
        parser.add_argument('--sgld-stepsize', type=float, default=1.0)
        parser.add_argument('--sgld-noise', type=float, default=0.01)
        parser.add_argument('--ebm-weight', type=float, default=1.0)
        parser.add_argument('--xent-weight', type=float, default=1.0)
        parser.add_argument('--l2-energy', type=float, default=0.0) 
        parser.add_argument('--sgld-warmup', type=int, default=0)
        parser.add_argument('--sgld-decay', type=float, default=0.0)
        parser.add_argument('--sgld-thresh', type=float, default=1.1)
        parser.add_argument('--sgld-optim', type=str, default='sgd')
        parser.add_argument('--sgld-replay-correction', type=float, default=1.0)
        parser.add_argument('--sgld-weight-decay', type=float, default=1e-05)

    @classmethod
    def build_objective(cls, conf):
        return EBMLoss(
            sgld_buffer=conf['sgld_buffer'],
            sgld_reinit_p=conf['sgld_reinit_p'],
            sgld_stepsize=conf['sgld_stepsize'],
            sgld_steps=conf['sgld_steps'],
            sgld_noise=conf['sgld_noise'],
            ebm_weight=conf['ebm_weight'],
            xent_weight=conf['xent_weight'],
            l2_energy=conf['l2_energy'],
            sgld_warmup=conf['sgld_warmup'],
            sgld_decay=conf['sgld_decay'],
            sgld_thresh=conf['sgld_thresh'],
            sgld_optim=conf['sgld_optim'],
            sgld_replay_correction=conf['sgld_replay_correction'],
            sgld_weight_decay=conf['sgld_weight_decay'],
        )

    def __init__(
        self,
        sgld_buffer=10000,
        sgld_reinit_p=0.05,
        sgld_stepsize=1.0,
        sgld_steps=20,
        sgld_noise=0.1,
        ebm_weight=1.0,
        xent_weight=1.0,
        l2_energy=0.0,
        sgld_warmup=0.0,
        sgld_decay=0.0,
        sgld_thresh=0.001,
        sgld_optim='sgd',
        sgld_replay_correction=1.0,
        sgld_weight_decay=1e-05,
    ):
        super(EBMLoss, self).__init__()

        self.sgld_sampler = SGLDSampler(
            buffer_size=sgld_buffer,
            sgld_reinit_p=sgld_reinit_p,
            sgld_stepsize=sgld_stepsize,
            sgld_noise=sgld_noise,
            num_steps=sgld_steps,
            sgld_thresh=sgld_thresh,
            sgld_optim=sgld_optim,
            sgld_replay_correction=sgld_replay_correction,
            sgld_weight_decay=sgld_weight_decay,
        )
        self.energy = Energy()
        
        # All of this is scheduling the lfmmi weight
        self.ebm_weight = ebm_weight
        self.xent_weight = xent_weight
        self.warmup = sgld_warmup
        self.decay = sgld_decay
        self.num_warmup_updates = 0 # Init to 1
        self.num_decay_updates = 0
        self.l2_energy = l2_energy
        self.sgld_thresh = sgld_thresh

    def forward(self, model, sample, precomputed=None):
        losses = []
        B = sample.input.size(0)
        model_energy = partial(self.energy, model)
        targets = sample.target 
        if precomputed is not None:
            x = precomputed
        else:
            x = model(sample)[0]
        
        T = x.size(1)
      
        if (targets[0, 0] == -1 and self.ebm_weight > 0):  
            sample_energy = -x.logsumexp(-1).sum() 
            avg_sample_energy = sample_energy

            # Figure out the weight to use
            if self.warmup > 0 and self.num_warmup_updates < self.warmup: 
                slope = self.ebm_weight / float(self.warmup) 
                curr_weight = slope * self.num_warmup_updates
                self.num_warmup_updates += 1
            else:
                factor = math.exp(-self.decay * self.num_decay_updates)
                curr_weight = self.ebm_weight * factor  
                self.num_decay_updates += 1 

            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            generated_samples, k = self.sgld_sampler.update(
                self.sgld_sampler.sample_like(sample),
                model_energy,
                sample_energy=avg_sample_energy.data.item(),
            )
            for p in model.parameters():
                p.requires_grad = True
            model.train()
      
            generated_samples_mb = Samples(generated_samples, sample.metadata)   
            expected_energy = model_energy(generated_samples_mb)
        
            # The gradient is E_p[\nabla E] - \nabla E.
            # We negate the loss because we have to minimize a function instead
            # of maximizing it.
            loss_ebm = -(expected_energy - avg_sample_energy) / (B * T)
            print('Expected_E: {}'.format(expected_energy.data.item()), end=' ')
            print('E: {}'.format(avg_sample_energy.data.item()), end=' ')
            print('EBM: {}'.format(loss_ebm.data.item()), end=' ')
            print('Curr_Weight: {}'.format(curr_weight), end=' ')
            print('Num_steps: {}'.format(k), end=' ')
            loss_ebm *= curr_weight
            losses.append(loss_ebm)
                        
            if self.l2_energy > 0: 
                loss_l2 = self.l2_energy * (expected_energy ** 2 + avg_sample_energy ** 2)
                print('L2: {}'.format(loss_l2.data.item()), end=' ')
                losses.append(loss_l2)

        correct = None
        if targets[0, 0] != -1 and self.xent_weight > 0:
            lprobs = F.log_softmax(x, dim=-1)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            loss_xent = F.nll_loss(lprobs, sample.target.view(-1), reduction='mean')
            correct = torch.sum(lprobs.argmax(1) == sample.target.view(-1))
            print('XENT: {}'.format(loss_xent.data.item()), end=' ')
            loss_xent *= self.xent_weight
            losses.append(loss_xent) 
        
        loss = sum(losses)
        
        return loss, correct 



