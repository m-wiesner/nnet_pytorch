import torch
import torch.nn as nn
import torch.nn.functional as F
from .pychain.pychain.graph import ChainGraphBatch, ChainGraph
import pychain_C
import simplefst
from .pychain.pychain.chain import ChainFunction 
from .SGLDSampler import SGLDSampler
from functools import partial
from .LFMMIOnly import NumeratorFunction
from .L2 import L2
from collections import namedtuple
import math


Samples = namedtuple('Samples', ['input', 'metadata']) 


class Energy(nn.Module):
    def __init__(self, graph):
        super(Energy, self).__init__()
        self.graph = graph
    
    def forward(self, model, sample):
        den_graphs = ChainGraphBatch(self.graph, sample.input.size(0))
        x = model(sample)[0]
        T = x.size(1)
        x_lengths = torch.LongTensor([T] * x.size(0)).to(x.device)
        objf = ChainFunction.apply(model(sample)[0], x_lengths, den_graphs, 0.1)
        return -objf 


class SequenceEBMLoss(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--denom-graph', required=True) 
        parser.add_argument('--sgld-steps', type=int, default=20)
        parser.add_argument('--sgld-buffer', type=int, default=10000)
        parser.add_argument('--sgld-reinit-p', type=float, default=0.05)
        parser.add_argument('--sgld-stepsize', type=float, default=1.0)
        parser.add_argument('--sgld-noise', type=float, default=0.01)
        parser.add_argument('--ebm-weight', type=float, default=1.0)
        parser.add_argument('--lfmmi-weight', type=float, default=1.0)
        parser.add_argument('--l2-energy', type=float, default=0.0) 
        parser.add_argument('--sgld-warmup', type=int, default=0)
        parser.add_argument('--sgld-decay', type=float, default=0.0)
        parser.add_argument('--sgld-thresh', type=float, default=1.1)
        parser.add_argument('--sgld-optim', type=str, default='sgd')
        parser.add_argument('--sgld-replay-correction', type=float, default=1.0)
        parser.add_argument('--sgld-weight-decay', type=float, default=1e-05)
        parser.add_argument('--sgld-max-steps', type=int, default=150)

    @classmethod
    def build_objective(cls, conf):
        return SequenceEBMLoss(
            conf['denom_graph'],
            sgld_buffer=conf['sgld_buffer'],
            sgld_reinit_p=conf['sgld_reinit_p'],
            sgld_stepsize=conf['sgld_stepsize'],
            sgld_steps=conf['sgld_steps'],
            sgld_noise=conf['sgld_noise'],
            ebm_weight=conf['ebm_weight'],
            lfmmi_weight=conf['lfmmi_weight'],
            l2_energy=conf['l2_energy'],
            sgld_warmup=conf['sgld_warmup'],
            sgld_decay=conf['sgld_decay'],
            sgld_thresh=conf['sgld_thresh'],
            sgld_optim=conf['sgld_optim'],
            sgld_replay_correction=conf['sgld_replay_correction'],
            sgld_weight_decay=conf['sgld_weight_decay'],
            sgld_max_steps=conf['sgld_max_steps'],
        )

    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration = None):
        return {
            'warmup': s1['warmup'],
            'decay': s1['decay'],
            'num_warmup_updates': int(s1['num_warmup_updates'] + fraction * s2['num_warmup_updates']),
            'num_decay_updates': int(s1['num_decay_updates'] + fraction * s2['num_decay_updates']),
            'sampler': SGLDSampler.add_state_dict(
                s1['sampler'], s2['sampler'], fraction, iteration=iteration, 
            ),
        }


    def __init__(
        self, den_graph,
        sgld_buffer=10000,
        sgld_reinit_p=0.05,
        sgld_stepsize=1.0,
        sgld_steps=20,
        sgld_noise=0.1,
        ebm_weight=1.0,
        lfmmi_weight=1.0,
        l2_energy=0.0,
        sgld_warmup=0.0,
        sgld_decay=0.0,
        sgld_thresh=0.001,
        sgld_optim='sgd',
        sgld_replay_correction=1.0,
        sgld_weight_decay=1e-05,
        sgld_max_steps=150,
    ):
        super(SequenceEBMLoss, self).__init__()
        self.den_graph = ChainGraph(
            fst=simplefst.StdVectorFst.read(den_graph),
        )

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
            sgld_max_steps=sgld_max_steps,
        )
        self.energy = Energy(self.den_graph)
        # All of this is scheduling the lfmmi weight
        self.ebm_weight = ebm_weight
        self.lfmmi_weight = lfmmi_weight
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
        x_lengths = torch.LongTensor([T] * B).to(x.device)
        den_graphs = ChainGraphBatch(self.den_graph, sample.input.size(0))
        sample_energy = -ChainFunction.apply(x, x_lengths, den_graphs, 0.1)
        avg_sample_energy = sample_energy
      
        if (targets[0, 0] == -1 and self.ebm_weight > 0):  
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

        if targets[0, 0] != -1 and self.lfmmi_weight > 0:
            num_objf = -NumeratorFunction.apply(x, sample.target)
            loss_lfmmi = (num_objf - sample_energy) / (B * T)  
            print('LFMMI: {}'.format(loss_lfmmi.data.item()), end=' ')
            loss_lfmmi *= self.lfmmi_weight
            losses.append(loss_lfmmi) 
        
        loss = sum(losses)
        
        return loss, None 

    def state_dict(self):
        return {
            'warmup': self.warmup,
            'decay': self.decay,
            'num_warmup_updates': self.num_warmup_updates,
            'num_decay_updates': self.num_decay_updates,
            'sampler': self.sgld_sampler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.sgld_sampler.load_state_dict(state_dict['sampler'])
        self.warmup = state_dict['warmup']
        self.decay = state_dict['decay']
        self.num_warmup_updates = state_dict['num_warmup_updates']
        self.num_decay_updates = state_dict['num_decay_updates']

    def generate_from_buffer(self):
        return self.sgld_sampler.buffer
    
    def generate_from_model(self, model,
        bs=32, cw=65, dim=64, left_context=10, right_context=5, device='cpu',
    ):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        model_energy = partial(self.energy, model)
        x = torch.FloatTensor(bs, cw, dim).uniform_(-1, 1) 
        x = x.to(device)
        return self.sgld_sampler.update_generator(
            self.sgld_sampler.sample_like(
                Samples(
                    x, 
                    {
                        'left_context': left_context,
                        'right_context': right_context
                    }
                )
            ),
            model_energy,
            sample_energy=-100.0,
        )


