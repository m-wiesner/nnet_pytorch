from transformers import HubertForCTC
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Hubert(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--hubert-mdl-name',
            type=str,
            default="facebook/hubert-large-ll60k"
        )
        parser.add_argument('--hubert-freeze-feat-extractor', action='store_true') 
        parser.add_argument('--hubert-subsampling', type=int, default=320)
        parser.add_argument('--hubert-single-layer', action='store_true')

    @classmethod
    def build_model(cls, conf):
        return Hubert(
            conf['num_targets'],
            mdl_name=conf.get('hubert_mdl_name', "facebook/hubert-large-ll60k"),
            freeze=conf.get('hubert_freeze_feat-extractor', False),
            subsampling=conf.get('hubert_subsampling', 320),
            single_layer=conf.get('hubert_single_layer', False),
        )

    def __init__(self, num_classes,
        mdl_name="facebook/hubert-large-ll60k",
        freeze=False,
        subsampling=320,
        single_layer=False,
    ):
        super(Hubert, self).__init__()
        self.single_layer = single_layer
        self.odim = num_classes
        self.freeze = freeze
        self.mdl_name = mdl_name
        self.subsampling = subsampling
        self.hubert = HubertForCTC.from_pretrained(mdl_name).hubert
        if self.freeze:
            self.hubert.feature_extractor._freeze_parameters()

        if not single_layer:
            self.downsample_linear = nn.Linear(1024, 512)
            self.lrelu = nn.LeakyReLU(0.2)
            self.linear = nn.Linear(512, num_classes)
        else:
            self.linear = nn.Linear(1024, num_classes)

    def forward(self, sample):
        x = self.hubert(sample.input.squeeze(-1)).last_hidden_state
        if not self.single_layer:
            x = self.downsample_linear(x)
            x = self.lrelu(x)

        # Downsampling in this model is from raw audio sampled at 16kHz and
        # and downsampled by a factor of 320
        left_context = sample.metadata['left_context']
        right_context = sample.metadata['right_context']
        cw = sample.input.size(1) - (left_context + right_context)
        cw_subsampled = len(range(0, cw, self.subsampling))
        output1 = F.adaptive_avg_pool1d(x.transpose(1,2), cw_subsampled).transpose(1, 2)
        output0 = self.linear(output1)
        return (output0, output1)


class ChainHubert(Hubert):
    @classmethod
    def build_model(cls, conf):
        return ChainHubert(
            conf['num_targets'],
            mdl_name=conf.get('hubert_mdl_name', "facebook/hubert-large-ll60k"),
            freeze=conf.get('hubert_freeze_feat-extractor', False),
            subsampling=conf.get('hubert_subsampling', 320),
            single_layer=conf.get('hubert_single_layer', False),
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefinal_xent = nn.Linear(
            512, 192,
        )
        self.xent_norm = nn.LayerNorm(192)
        self.xent_layer = nn.Linear(192, self.odim)
    
    def forward(self, sample):
        output, x = super().forward(sample)
        if self.training:
            x = self.lrelu(self.prefinal_xent(x))
            x = self.xent_norm(x)
            x = self.xent_layer(x)
        
        return (output, x)
