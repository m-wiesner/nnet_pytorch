import torch
import torch.nn as nn
import torch.nn.functional as F
from .pychain.pychain.graph import ChainGraphBatch, ChainGraph
import pychain_C
import simplefst
from .pychain.pychain.chain import ChainFunction 
from .SGLDSampler import SGLDSampler
from functools import partial
import random
from .LFMMIOnly import NumeratorFunction
from .L2 import L2
from collections import namedtuple
import math


Samples = namedtuple('Samples', ['input', 'metadata']) 


class Energy(nn.Module):
    def __init__(self, graph, leaky_hmm=0.1):
        super(Energy, self).__init__()
        self.graph = graph
        self.leaky_hmm = leaky_hmm
    
    def forward(self, model, sample):
        den_graphs = ChainGraphBatch(self.graph, sample.input.size(0))
        x = model(sample)[0]
        T = x.size(1)
        x_lengths = torch.LongTensor([T] * x.size(0)).to(x.device)
        objf = ChainFunction.apply(x, x_lengths, den_graphs, self.leaky_hmm)
        return -objf 


class TargetEnergy(nn.Module):
    def __init__(self):
        super(TargetEnergy,self).__init__()

    def forward(self, model, sample, target=[]):
        x = model(sample)[0].clamp(-30, 30)
        target = torch.LongTensor(target).to(x.device).unsqueeze(2)
        return -x.gather(2, target).sum()


class SequenceEBMLoss(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--denom-graph', required=True) 
        parser.add_argument('--leaky-hmm', type=float, default=0.1)
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
        parser.add_argument('--sgld-init-real-decay', type=float, default=0.0)
        parser.add_argument('--sgld-clip', type=float, default=1.0)
        parser.add_argument('--sgld-init-val', type=float, default=1.0)
        parser.add_argument('--sgld-epsilon', type=float, default=1e-04) 
        parser.add_argument('--ebm-tgt', type=str, default=None)
        parser.add_argument('--ebm-type', type=str, default='uncond', choices=['uncond', 'cond'])
        parser.add_argument('--ebm-joint', action='store_true')

    @classmethod
    def build_objective(cls, conf):
        if conf.get('ebm_type', 'uncond') == 'cond':
            if conf.get('ebm_tgt', None) is None:
                print("ERROR: Conditional EBM model needs to have targets specified.") 
                sys.exit(1)

        return SequenceEBMLoss(
            conf['denom_graph'],
            leaky_hmm=conf.get('leaky_hmm', 0.1),
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
            sgld_init_real_decay=conf.get('sgld_init_real_decay', 0.0),
            sgld_clip=conf['sgld_clip'],
            sgld_init_val=conf.get('sgld_init_val', 1.0),
            sgld_epsilon=conf.get('sgld_epsilon', 1e-04),
            ebm_type=conf.get('ebm_type', 'uncond'),
            ebm_tgt=conf.get('ebm_tgt', None),
            joint_model=conf.get('ebm_joint', False),
        )

    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration = None):
        return {
            'warmup': s1['warmup'],
            'decay': s1['decay'],
            'init_real_decay': s1['init_real_decay'],
            'num_warmup_updates': int(s1['num_warmup_updates'] + fraction * s2['num_warmup_updates']),
            'num_decay_updates': int(s1['num_decay_updates'] + fraction * s2['num_decay_updates']),
            'num_decay_real_updates': int(s1['num_decay_real_updates'] + fraction * s2['num_decay_real_updates']),
            'sampler': SGLDSampler.add_state_dict(
                s1['sampler'], s2['sampler'], fraction, iteration=iteration, 
            ),
        }


    def __init__(
        self, den_graph,
        leaky_hmm=0.1,
        sgld_init_val=1.0,
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
        sgld_init_real_decay=0.0,
        sgld_clip=1.0,
        sgld_epsilon=1e-04,
        ebm_type='uncond',
        ebm_tgt=None,
        joint_model=False,
    ):
        super(SequenceEBMLoss, self).__init__()
        self.den_graph = ChainGraph(
            fst=simplefst.StdVectorFst.read(den_graph),
        )
        self.leaky_hmm = leaky_hmm

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
            sgld_clip=sgld_clip,
            sgld_init_val=sgld_init_val,
            sgld_epsilon=sgld_epsilon,
        )
        # All of this is scheduling the lfmmi weight
        self.ebm_weight = ebm_weight
        self.lfmmi_weight = lfmmi_weight
        self.warmup = sgld_warmup
        self.decay = sgld_decay
        self.num_warmup_updates = 0 # Init to 1
        self.num_decay_updates = 0
        self.l2_energy = l2_energy
        self.sgld_thresh = sgld_thresh
        self.num_decay_real_updates = 0
        self.init_real_decay = sgld_init_real_decay
        self.ebm_type = ebm_type
        if ebm_type == 'cond':
            self.ebm_tgt = ebm_tgt
            self.tgt_lines = {}
            with open(ebm_tgt) as f:
                i = 0
                start = f.tell()
                line = f.readline()
                while line: 
                    self.tgt_lines[i] = (start, len(line.split())-1)
                    start = f.tell()  
                    line = f.readline()
                    i += 1
            self.cond_energy = 0.0
        else:
            self.ebm_tgt = None
        self.energy = Energy(self.den_graph, leaky_hmm=self.leaky_hmm)
        self.joint_model = joint_model

    def forward(self, model, sample, precomputed=None):
        losses = []
        B = sample.input.size(0)
        targets = sample.target 
        if precomputed is not None:
            x = precomputed
        else:
            x = model(sample)[0]
        
        T = x.size(1)
        model_energy = partial(self.energy, model)
        x_lengths = torch.LongTensor([T] * B).to(x.device)
        den_graphs = ChainGraphBatch(self.den_graph, sample.input.size(0))
        sample_energy = -ChainFunction.apply(x, x_lengths, den_graphs, self.leaky_hmm)
        avg_sample_energy = sample_energy
     
        is_unsup = (targets[0, 0] == -1) 
        if (is_unsup and self.ebm_weight > 0) or (self.joint_model and self.ebm_weight > 0):  
            if self.ebm_type == 'cond':
                tgts = self.sample_targets(B, T)
                sampling_energy = TargetEnergy()            
                sampling_model_energy = partial(sampling_energy.forward, model, target=tgts)
                ground_truth_energy = self.cond_energy * (B * T)
            else:
                sampling_model_energy = model_energy
                ground_truth_energy = avg_sample_energy.data.item()

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

            # For contrastive divergence initializing with real samples
            real_weight = 1.0 - math.exp(
                -self.init_real_decay * self.num_decay_real_updates
            )
            self.num_decay_real_updates += 1
            
            generated_samples, k = self.sgld_sampler.update(
                self.sgld_sampler.sample_like(sample, alpha=real_weight),
                sampling_model_energy,
                sample_energy=ground_truth_energy,
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
            print('Real_Weight: {}'.format(real_weight), end=' ')
            loss_ebm *= curr_weight
            losses.append(loss_ebm)
                        
            if self.l2_energy > 0: 
                loss_l2 = self.l2_energy * (expected_energy ** 2 + avg_sample_energy ** 2)
                print('L2: {}'.format(loss_l2.data.item()), end=' ')
                losses.append(loss_l2)

            if self.joint_model:
                print()

        if targets[0, 0] != -1 and self.lfmmi_weight > 0:
            num_objf = -NumeratorFunction.apply(x, sample.target)
            self.cond_energy = num_objf.data.item() / (B * T) 
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
            'init_real_decay': self.init_real_decay,
            'num_warmup_updates': self.num_warmup_updates,
            'num_decay_updates': self.num_decay_updates,
            'num_decay_real_updates': self.num_decay_real_updates,
            'sampler': self.sgld_sampler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.sgld_sampler.load_state_dict(state_dict['sampler'])
        self.warmup = state_dict['warmup']
        self.decay = state_dict['decay']
        self.num_warmup_updates = state_dict['num_warmup_updates']
        self.num_decay_updates = state_dict['num_decay_updates']
        self.num_decay_real_updates = state_dict.get('num_decay_real_updates', 0.0) # 0.0 for back compatibility 
        self.init_real_decay = state_dict.get('init_real_decay', 0.0) # 0.0 for back compatibility

    def generate_from_buffer(self):
        return self.sgld_sampler.buffer
    
    def generate_from_model(self, model,
        bs=32, cw=65, dim=64, left_context=10, right_context=5, device='cpu',
        target=None,
    ):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        if target is not None:
            energy = TargetEnergy()
            model_energy = partial(energy, model, target=target) 
        else:
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

    def decorrupt(self, model, sample, num_steps=None):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        model_energy = partial(self.energy, model)
        if num_steps is not None:
            self.sgld_sampler.max_steps = num_steps
        return self.sgld_sampler.update_generator(
            (
                sample.input,
                torch.randint(0, 1, (0,)),
                sample.metadata,
                torch.zeros(sample.input.size(0)).to(sample.input.device),
            ),
            model_energy,
            sample_energy=-100.0,
        )

    def sample_targets(self, bs, length):
        f = open(self.ebm_tgt)
        offsets = random.choices(self.tgt_lines, k=bs)
        tgts = []
        for offset in offsets:
            idx = 1 + int((random.random() - 1e-09) * (offset[1] - (length-1)))
            f.seek(offset[0])
            # The first token is the uttid
            tgts.append(
                [int(pdf) for pdf in f.readline().split()[idx: idx + length]]
            )
        f.close()
        return tgts            

