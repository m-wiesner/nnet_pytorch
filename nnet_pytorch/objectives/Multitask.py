# Copyright 2021
# Apache 2.0

import torch
import torch.nn as nn
from .L2 import L2
from .CrossEntropy import CrossEntropy
from .LFMMIOnly import ChainLoss as LFMMI
from .InfoNCEOnly import InfoNCELoss as InfoNCE
from .EnergyObjective import EnergyLoss
from .InfoNCE2pass import InfoNCELoss as InfoNCE2pass


OBJECTIVES = {
    'CrossEntropy': CrossEntropy,
    'LFMMIOnly': LFMMI,
    'Energy': EnergyLoss,
    'InfoNCE': InfoNCE,
    'InfoNCE2pass': InfoNCE2pass,
    'L2': L2,
}


class MultitaskLoss(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--multitask-losses', type=str, required=True,
            default="[('LFMMIOnly', 1.0, 0), ('L2', 0.0001, 0)]", 
        )
        losses_args, extra = parser.parse_known_args()
        for l in eval(losses_args.multitask_losses):
            OBJECTIVES[l[0]].add_args(parser)

    @classmethod
    def build_objective(cls, conf):
        losses = {}
        for loss in eval(conf['multitask_losses']): 
            name, weight, branch = loss[0], loss[1], loss[2]
            loss_class = OBJECTIVES[name].build_objective(conf)
            losses[name] = {'class': loss_class, 'weight': weight, 'branch': branch} 
        return MultitaskLoss(losses)

    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration=None):
        new_losses = {} 
        for name, loss in s1.items():
            state_dict_class = OBJECTIVES[name]
            new_losses[name] = state_dict_class.add_state_dict(
                loss, s2[name], fraction, iteration=iteration
            )
        return new_losses
             
    def __init__(self, losses):
        super(MultitaskLoss, self).__init__() 
        self.losses = nn.ModuleDict({l: losses[l]['class'] for l in losses})
        self.weights, self.branches = {}, {}
        max_branch = 0
        for l, v in losses.items():
            self.weights[l], self.branches[l] = v['weight'], v['branch']
            max_branch = v['branch'] if v['branch'] > max_branch else max_branch
        
        # The branches are 0 indexed
        self.num_branches = 1 + max_branch
   
    def forward(self, model, sample, precomputed=None):
        if precomputed is None:
            x = model(sample)
        else:
            x = precomputed
        if len(x) < self.num_branches:
            raise ValueError('The number of model branches does not match the '
                ' number of branches used in the multitask loss')
      
        # See which objfs to use 
        objf_names = sample.metadata.get('objf_names', None)
        if objf_names is None:
            objf_names = self.losses.keys() 
        losses = []
        for n in objf_names:
            loss = self.losses[n]
            weight, branch = self.weights[n], self.branches[n]
            if weight > 0:
                loss_value, _ = loss(model, sample, precomputed=x[branch])
                print(f'{n}: {loss_value.data.item()}', end=' ') 
                losses.append(weight * loss_value)
        return sum(losses), None

    def state_dict(self):
        return { n: v.state_dict() for n, v in self.losses.items()}
    
    def load_state_dict(self, state_dict):
        for n, v in state_dict.items():
            self.losses[n].load_state_dict(v)
    
    def decorrupt(self, *args, **kwargs):
        for n, v in self.losses.items():
            if hasattr(v, 'decorrupt'):
                return v.decorrupt(*args, **kwargs)
        raise RuntimeError("No objective function was capable of decorruption")

    def generate_from_buffer(self):
        for n, v in self.losses.items():
            if hasattr(v, 'generate_from_buffer'):
                return v.generate_from_buffer()
        raise RuntimeError("No objective function was capable of generating "
            "from a buffer")
