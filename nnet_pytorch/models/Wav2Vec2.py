from transformers import Wav2Vec2ForCTC
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Wav2Vec2(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--wav2vec2-mdl-name',
            type=str,
            default="facebook/wav2vec2-large-100k-voxpopuli"
        )
        parser.add_argument('--wav2vec2-freeze-feat-extractor', action='store_true') 
        parser.add_argument('--wav2vec2-subsampling', type=int, default=320)
        parser.add_argument('--wav2vec2-single-layer', action='store_true')

    @classmethod
    def build_model(cls, conf):
        return Wav2Vec2(
            conf['num_targets'],
            mdl_name=conf.get('wav2vec2_mdl_name', "facebook/wav2vec2-large-100k-voxpopuli"),
            freeze=conf.get('wav2vec2_freeze_feat-extractor', False),
            subsampling=conf.get('wav2vec2_subsampling', 320),
            single_layer=conf.get('wav2vec2_single_layer', False),
        )

    def __init__(self, num_classes,
        mdl_name="facebook/wav2vec2-large-100k-voxpopuli",
        freeze=False,
        subsampling=320,
        single_layer=False,
    ):
        super(Wav2Vec2, self).__init__()
        self.single_layer = single_layer
        self.odim = num_classes
        self.freeze = freeze
        self.mdl_name = mdl_name
        self.subsampling = subsampling
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained(mdl_name).wav2vec2
        if self.freeze:
            self.wav2vec2.feature_extractor._freeze_parameters()

        if not single_layer:
            self.downsample_linear = nn.Linear(1024, 512)
            self.lrelu = nn.LeakyReLU(0.2)
            self.linear = nn.Linear(512, num_classes)
        else:
            self.linear = nn.Linear(1024, num_classes)

    def forward(self, sample):
        x = self.wav2vec2(sample.input.squeeze(-1)).last_hidden_state
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


class ChainWav2Vec2(Wav2Vec2):
    @classmethod
    def build_model(cls, conf):
        return ChainWav2Vec2(
            conf['num_targets'],
            mdl_name=conf.get('wav2vec2_mdl_name', "facebook/wav2vec2-large-100k-voxpopuli"),
            freeze=conf.get('wav2vec2_freeze_feat-extractor', False),
            subsampling=conf.get('wav2vec2_subsampling', 320),
            single_layer=conf.get('wav2vec2_single_layer', False),
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
