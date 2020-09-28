import torch.nn.functional as F
import torch


class CrossEntropy(torch.nn.Module):
    @staticmethod
    def add_args(parser):
        pass
   
    @classmethod 
    def build_objective(cls, conf):
        return CrossEntropy()
         
    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration=None):
        return s1

    def __init__(self):
        super(CrossEntropy, self).__init__()
    
    def forward(self, model, sample, precomputed=None):
        if precomputed is not None:
            output = precomputed
        else:
            output = model(sample)[0]
        
        lprobs = F.log_softmax(output, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss = F.nll_loss(lprobs, sample.target.view(-1), reduction='mean')
        correct = torch.sum(lprobs.argmax(1) == sample.target.view(-1))
        return loss, correct


         
        
