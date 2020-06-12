import torch
import torch.nn.functional as F
import numpy as np


class TDNN(torch.nn.Module):
    '''
        Kaldi TDNN style encoder implemented as convolutions
    '''
    @staticmethod
    def add_args(parser):
        parser.add_argument('--tdnn-hdim', type=int, default=625)
        parser.add_argument('--tdnn-num-layers', type=int, default=10)
        parser.add_argument('--tdnn-dropout', type=float, default=0.1)
        parser.add_argument('--tdnn-prefinal-dim', type=int, default=192)
     
    @classmethod
    def build_model(cls, conf):
        model = TDNN(
            conf['idim'], conf['num_targets'],
            odims=[conf['tdnn_hdim'] for i in range(conf['tdnn_num_layers'])],
            dropout=conf['tdnn_dropout'],
            prefinal_affine_dim=conf['tdnn_prefinal_dim'],
            subsample=conf['subsample'],
            batch_norm_dropout=False
        )   
        return model
    
    def __init__(
        self, idim, odim,
        odims=[625, 625, 625, 625, 625, 625],
        prefinal_affine_dim=625,
        offsets=[
            [0], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], 
            [-3, 0, 3], [-3, 0, 3], [-3, 0, 3], [-3, 0, 3],
            [-3, 0, 3], [-3, 0, 3], [-3, 0, 3], [-3, 0, 3],
            [-3, 0, 3], [-3, 0, 3], [-3, 0, 3], [-3, 0, 3],
        ], nonlin=F.relu, dropout=0.1, subsample=1, batch_norm_dropout=False,
    ):
        super().__init__()
        
        # Proper TDNN layers
        odims_ = list(odims)
        odims_.insert(0, idim)
        self.batch_norm_dropout = False
        self.dropout = dropout
        self.nonlin = nonlin
        self.subsample = subsample
        self.tdnn = torch.nn.ModuleList()
        self.batchnorm = torch.nn.ModuleList()
        i = 0
        #for offs in offsets:
        for layer in range(len(odims)):
            offs = offsets[layer]
            # Calculate dilations
            if len(offs) > 1:
                dilations = np.diff(offs)
                if np.all(dilations == dilations[0]):
                    dil = dilations[0]
                    pad = max(offs)
                else:
                    sys.exit("Not non-uniform offsets not implemented")
            else:
                dil = 1
                pad = 0
            self.tdnn.append(
                torch.nn.Conv1d(
                    odims_[i], odims_[i+1],
                    len(offs), stride=1, dilation=dil, padding=pad
                )
            )
            self.batchnorm.append(
                torch.nn.BatchNorm1d(odims_[i+1], eps=1e-03, affine=False)
            )
            i += 1

        # Last few layers
        self.prefinal_affine = torch.nn.Conv1d(
            odims_[i], prefinal_affine_dim, 1,
            stride=1, dilation=1, bias=True, padding=0
        )
        self.batchnorm.append(
            torch.nn.BatchNorm1d(
                prefinal_affine_dim, eps=1e-03, affine=False
            )
        )
        self.final_affine = torch.nn.Linear(
            prefinal_affine_dim, odim,
        )

    def forward(self, sample):
        xs_pad = sample.input
        left_context = sample.metadata['left_context']
        right_context = sample.metadata['right_context']
        
        # Just to get shape right for convolutions
        xs_pad = xs_pad.transpose(1, 2)
       
        # Basic pattern is (tdnn, relu, batchnorm, dropout) x num_layers 
        for tdnn, batchnorm in zip(self.tdnn, self.batchnorm[:-1]):
            xs_pad = self.nonlin(tdnn(xs_pad))
            if not self.batch_norm_dropout: 
                xs_pad = batchnorm(xs_pad)
                xs_pad = F.dropout(xs_pad, p=self.dropout, training=self.training)
      
        # A few final layers
        end_idx = xs_pad.size(2) if right_context == 0 else -right_context
        output2 = xs_pad.transpose(1, 2)[:, left_context:end_idx:self.subsample, :]
        xs_pad = self.nonlin(self.prefinal_affine(xs_pad))
        if not self.batch_norm_dropout:
            xs_pad = self.batchnorm[-1](xs_pad)
        
        # This is basically just glue
        output = xs_pad.transpose(1, 2)
        output = self.final_affine(output)
        return (
            output[:, left_context:end_idx:self.subsample, :],
            output2,
        )


class ChainTDNN(TDNN):
    @classmethod
    def build_model(cls, conf):
        model = ChainTDNN(
            conf['idim'], conf['num_targets'],
            odims=[conf['tdnn_hdim'] for i in range(conf['tdnn_num_layers'])],
            dropout=conf['tdnn_dropout'],
            prefinal_affine_dim=conf['tdnn_prefinal_dim'],
            subsample=conf['subsample'],
        )   
        return model

    def __init__(
        self, idim, odim,
        odims=[625, 625, 625, 625, 625, 625],
        prefinal_affine_dim=625,
        offsets=[
            [0], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], 
            [-3, 0, 3], [-3, 0, 3], [-3, 0, 3], [-3, 0, 3],
            [-3, 0, 3], [-3, 0, 3], [-3, 0, 3], [-3, 0, 3],
            [-3, 0, 3], [-3, 0, 3], [-3, 0, 3], [-3, 0, 3],
        ], nonlin=F.relu, dropout=0.1, subsample=1,
    ):
        super().__init__(
            idim, odim, odims, prefinal_affine_dim,
            offsets, nonlin, dropout, subsample
        )
        self.prefinal_xent = torch.nn.Linear(
            odims[-1],
            prefinal_affine_dim,
        )
        self.xent_batchnorm = torch.nn.BatchNorm1d(
            prefinal_affine_dim,
            eps=1e-03, affine=False
        )
        self.xent_layer = torch.nn.Linear(prefinal_affine_dim, odim)
    
    def forward(self, xs_pad):
        output, xs_pad = super().forward(xs_pad)
        if self.training:
            xs_pad = self.nonlin(self.prefinal_xent(xs_pad))
            if not self.batch_norm_dropout:
                xs_pad = self.xent_batchnorm(xs_pad.transpose(1, 2)).transpose(1, 2)
            xs_pad = self.xent_layer(xs_pad)
        return output, xs_pad 
