import torch.nn.functional as F
import torch


class L2(torch.nn.Module):
    @staticmethod
    def add_args(parser):
        pass

    @classmethod
    def build_objective(cls, conf):
        return L2(conf)
    
    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration=None):
        return s1
    
    def __init__(self):
        super(L2, self).__init__()
    
    def forward(self, model, sample, precomputed=None):
        if precomputed is not None:
            x = precomputed
        else:
            x = model(sample)[0]
        
        loss = ((x ** 2).sum()) / (x.size(0) * x.size(1))
        return loss, None 
