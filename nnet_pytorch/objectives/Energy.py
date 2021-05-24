import torch
import torch.nn as nn
from .pychain.pychain.graph import ChainGraphBatch, ChainGraph
import pychain_C
import simplefst
from .pychain.pychain.chain import ChainFunction


class Energy(nn.Module):
    @staticmethod
    def add_args(parser):
        pass

    @classmethod
    def build_energy(cls, conf):
        return Energy()

    def __init__(self):
        super(Energy, self).__init__()
    
    def forward(self, model, sample, precomputed=None):
        '''
            Defines the interface for an energy of a sample given a model.
            There are multiple ways to define the energy, but they require
            a model output.
        '''
        raise NotImplementedError
        return energy


class ClassificationEnergy(Energy):
    '''
        Defines the energy from a simple classification problem. In the event
        that the output is a sequence, it treats each output independently and
        normalizes by sequence length so that longer sequences don't
        automatically have higher energy.
    '''
    @classmethod
    def build_energy(cls, conf):
        return ClassificationEnergy()

    def forward(self, model, sample, precomputed=None):
        if precomputed is not None:
            x = precomputed
        else:
            x = forward_no_grad(model, sample)[0]
        
        T = x.size(1)
        energy = -x.logsumexp(-1).sum() / T
        return energy


class LFMMIEnergy(Energy):
    '''
        Defines the energy using the LFMMI denominator graph. The energy is
        the forward score of the input sample through the denominator graph.
    '''
    @staticmethod
    def add_args(parser):
        parser.add_argument('--energy-denom-graph', required=True)
        parser.add_argument('--energy-leaky-hmm', type=float, default=0.1)

    @classmethod
    def build_energy(cls, conf):
        if 'energy_denom_graph' not in conf:
            raise ValueError('User did not specify energy_denom_graph')
             
        return LFMMIEnergy(
            conf['energy_denom_graph'],
            leaky_hmm=conf.get('energy_leaky_hmm', 0.1),
        )

    def __init__(self, graph, leaky_hmm=0.1):
        super(LFMMIEnergy, self).__init__()
        self.graph = ChainGraph(
            fst=simplefst.StdVectorFst.read(graph),
        )
        self.leaky_hmm = leaky_hmm
        
    def forward(self, model, sample, precomputed=None):
        den_graphs = ChainGraphBatch(self.graph, sample.input.size(0))
        if precomputed is not None:
            x = precomputed
        else:
            x = forward_no_grad(model, sample)[0]
        
        T = x.size(1)
        x_lengths = torch.LongTensor([T] * x.size(0)).to(x.device)
        energy = -ChainFunction.apply(x, x_lengths, den_graphs, self.leaky_hmm, clamp=False)
        return energy    
    

class TargetEnergy(Energy):
    '''
        Defines the conditional energy of an input given a fixed target. The
        energy is the sum of the values of outputs nodes y_t for each time
        t=0 to T-1, where T is the length of the sequence.
    '''
    def forward(self, model, sample, precomputed=None, target=[], reduction='sum'):
        if precomputed is not None:
            x = precomputed
        else:
            x = forward_no_grad(model, sample)[0]
        target = torch.LongTensor(target).to(x.device).unsqueeze(2)
        energy = -x.gather(2, target)
        if reduction == 'sum':
            return energy.sum()
        elif reduction == 'none':
            return energy.sum(dim=1)
        elif reduction == 'mean':
            return energy.sum() / energy.size(0)


def no_grad(fun):
    '''
        Decorator for turning off gradient tracking on the model, but not on
        the input sample.
    '''
    def wrapper(*args):
        model = args[0]
        sample = args[1]
        for p in model.parameters():
            p.requires_grad = False 
        x = fun(model, sample)
        for p in model.parameters():
            p.requires_grad = True
        return x
    return wrapper


@no_grad
def forward_no_grad(model, sample):
    return model(sample)
      
