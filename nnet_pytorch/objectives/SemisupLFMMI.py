import torch
import torch.nn as nn
import torch.nn.functional as F
from .L2 import L2
from .CrossEntropy import CrossEntropy 
from .LFMMI_EBM import SequenceEBMLoss as SeqEBM


class ChainLoss(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--unsup-weight', type=float, default=1.0)
        parser.add_argument('--xent-reg', type=float, default=0.2)
        parser.add_argument('--l2-reg', type=float, default=0.00025)
        for m in [L2, CrossEntropy, SeqEBM]:
            m.add_args(parser) 

    @classmethod
    def build_objective(cls, conf):
        seq_ebm = SeqEBM.build_objective(conf)
        return ChainLoss(
            seq_ebm,
            unsup_weight=conf['unsup_weight'],
            xent_reg=conf['xent_reg'],
            l2_reg=conf['l2_reg'],
        )

    def __init__(
        self, seq_ebm, xent_reg=0.2, l2_reg=0.00025, unsup_weight=1.0,
    ):
        super(ChainLoss, self).__init__()
        self.seq_ebm = seq_ebm
        self.xent = CrossEntropy()
        self.l2 = L2()
        
        self.l2_reg = l2_reg
        self.xent_reg = xent_reg
        self.unsup_weight = unsup_weight

    def forward(self, model, sample):
        chain_output = model(sample)
        is_unsup = sample.target[0, 0] == -1 
        losses = [] 
        correct = None
        # SeqEBM
        loss_seqebm, _ = self.seq_ebm(
            model,
            sample,
            precomputed=chain_output[0],
        )
        losses.append(loss_seqebm)
        
        # XENT
        if not is_unsup and self.xent_reg > 0:
            loss_xent, correct = self.xent(
                model,
                sample,
                precomputed=chain_output[1],
            )
            loss_xent *= self.xent_reg
            print('XENT: {}'.format(loss_xent.data.item()), end=' ')
            losses.append(loss_xent)
        
        # L2
        if self.l2_reg > 0 and not is_unsup:
            loss_l2, _ = self.l2(
                model,
                sample,
                precomputed=chain_output[0],
            )
            loss_l2 *= self.l2_reg
            print('L2: {}'.format(loss_l2.data.item()), end=' ')
            losses.append(loss_l2)

        loss = sum(losses)
        return loss, correct

    def state_dict(self):
        return {'seq_ebm': self.seq_ebm.state_dict()}

    def load_state_dict(self, state_dict):
        self.seq_ebm.load_state_dict(state_dict)
        
