import numpy as np
import torch
from data_utils import memmap_feats, get_targets
from bisect import bisect
from collections import namedtuple
from data_utils import memmap_feats, get_targets, load_cmvn, load_utt2spk, load_utt_subset, Minibatch
import kaldi_io
import os
import pickle
import random
from itertools import groupby


class HybridAsrDataset(object):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--skip-datadump', action='store_true')
        parser.add_argument('--perturb-type', type=str, default='none')
        parser.add_argument('--validation-spks', type=int, default=5)
        parser.add_argument('--utt-subset', default=None)
        parser.add_argument('--mean-var', default="(True, 'norm')")

    def __init__(self, datadir, targets, num_targets,
        dtype=np.float32, memmap_affix='.dat',
        left_context=10, right_context=3, chunk_width=1,
        batchsize=128, num_split=1, mean=True, var='norm',
        skip_datadump=True, validation=1, utt_subset=None, subsample=1,
        perturb_type='none',
    ):
        # Load CMVN
        cmvn_scp = os.path.sep.join((datadir, 'cmvn.scp'))
        self.perturb_type = perturb_type
        self.mean = mean
        self.var = var
        self.subsample = subsample
        self.spk2cmvn = load_cmvn(cmvn_scp)
        self.batchsize = batchsize
        
        # Load UTT2SPK
        utt2spk = os.path.sep.join((datadir, 'utt2spk'))
        with open(utt2spk) as f:
            self.utt2spk, self.spk2utt = load_utt2spk(f)
       
        # Get some held out speakers 
        self.heldout = set()
        if validation > 0:
            self.heldout = set(random.sample(self.spk2utt.keys(), validation))   
                
        # Dump memmapped features (Faster I/O and no egs creation)
        feats_scp = os.path.sep.join((datadir,'feats.scp'))
        f_memmap = feats_scp + memmap_affix
        metadata_path = os.path.sep.join((datadir,'mapped','metadata'))
        self.data_path = os.path.sep.join((datadir,'mapped','feats.dat'))
        if skip_datadump:
            print('Skipping data dump. Remove flag --skip-datadump to redo this step')
            utt_lengths = []
            offsets = []
            data_shape = []
            for n in range(1,1+num_split):
                with open(metadata_path + '.' + str(n) + '.pkl', 'rb') as f:
                    utt_lengths_n, offsets_n, data_shape_n = pickle.load(f)
                    utt_lengths.append(utt_lengths_n)
                    offsets.append(offsets_n)
                    data_shape.append(data_shape_n)
        else:
            utt_lengths, offsets, data_shape = memmap_feats(feats_scp, f_memmap)
            with open(feats_scp + '.pkl', 'bw') as f:
                pickle.dump([utt_lengths, offsets, data_shape], f) 
        
        self.data = [np.memmap(
            "{}.{}".format(self.data_path,n), dtype=dtype, shape=data_shape[n-1], mode='r'
        ) for n in range(1,1+num_split)]

        # Get some metadata
        self.utt_lengths = utt_lengths # list of dicts with (utt, len) (key, val) pairs
        self.offsets_dict = offsets # list of dicts with (utt, offset) (key, val) pairs
        self.utt_offsets = [sorted(offset_n.items(), key=lambda x: x[1]) for offset_n in offsets] # sorted list of utterances by offset
        self.offsets = [[u[1] for u in offset_n] for offset_n in self.utt_offsets] # list of sorted list of offsets
        self.data_shape = data_shape # list of shape of the whole data (per split)
                
        # Get targets. Dummy targets for utterances where labels are unknown
        with open(targets) as f:
            self.targets = get_targets(f)
        self.num_targets = num_targets # Number of output nodes in DNN
        self.left_context = left_context
        self.right_context = right_context
        self.chunk_width = chunk_width
        self.subsample_chunk_width = len(
            range(0, self.chunk_width, self.subsample)
        )

        # Get the subset of speakers we actually want to use (important for evaluation)
        if utt_subset is not None:
            with open(utt_subset) as f:
                self.utt_subset = load_utt_subset(f)
        else:
            self.utt_subset = [i for i in self.targets]

    def free_ram(self, split_idx):
        dtype = self.data[0].dtype
        del self.data[split_idx]
        self.data.insert(split_idx, np.memmap(
            "{}.{}".format(self.data_path,split_idx+1), dtype=dtype, shape=self.data_shape[split_idx], mode='r',
        )) # file indexes are 1-based while split_idx is 0-based
        print (len(self.data))

    def __getitem__(self, index):
        '''
            Retreive an object representing the data at the specified index.
            Returns a named tuple.
            The index argument must be a tuple specifying the data split index
            and the sample index within the data split.
            
            type(index) == tuple(int, int)
            output.input == np.array(
                [
                    [v1, v2, v3, ...],
                    [v4, v5, v6, ...],
                    ...,
                    [v7, v8, v9, ...]
                ]
            )
        '''
        split_idx, idx = index
        # Find which utterance the index belongs to
        utt_idx = max(0, bisect(self.offsets[split_idx], idx) - 1)
        utt_name, offset = self.utt_offsets[split_idx][utt_idx]
        # Check that the utterances has a target (successful alignment)
        if utt_name not in self.targets:
            return None 
        utt_length = self.utt_lengths[split_idx][utt_name]
        # Retrieve the appropriate target
        target_start = (idx - offset) // self.subsample
        target_end = min(
            len(self.targets[utt_name]),
            target_start + self.subsample_chunk_width,
        )
        
        target = self.targets[utt_name][target_start: target_end]
        # Get the lower and upper boundaries for the window
        # (left_context + index + right_context) 
        #lower_boundary = max(0, index - self.left_context)
        #upper_boundary = min(
        #    self.data_shape[0], index + self.right_context + self.chunk_width
        #)
        lower_boundary = max(offset, idx - self.left_context)
        upper_boundary = min(
            offset + utt_length, idx + self.right_context + self.chunk_width
        )

        x = np.array(self.data[split_idx][lower_boundary: upper_boundary, :])
                
        # Apply cmvn
        if self.mean or self.var:
            x = self.apply_cmvn(x, utt_name, mean=self.mean, var=self.var)

        # This is solving the edge case needed when the beginning
        # and end frames need to be padded with the appropriate contexts
        left_zero_pad = max(0, self.left_context - (idx - lower_boundary))
        right_zero_pad = max(0, self.right_context + (self.chunk_width - 1) - ((upper_boundary-1) - idx))
        if left_zero_pad > 0 or right_zero_pad > 0: 
            x = np.pad(
                x, ((left_zero_pad, right_zero_pad,), (0, 0)),
                mode='edge'
            )
                
        # Collect metadata
        metadata = {
            'name': utt_name,
            'index': idx,
            'offset': offset,
            'length': utt_length,
        }
        return Minibatch(x, target, metadata)

    def apply_cmvn(self, x, utt_name, mean=False, var='norm', max_val=32.0, min_val=-16.0):
        '''
            Apply the speaker-level cmvn to each window (x).
        '''
        if not mean and not var:
            return x
        if mean:
            x_ = x - self.spk2cmvn[self.utt2spk[utt_name]]['mu']
        else:
            x_ = x
        
        if var == 'var':
            x_ = x_ / np.sqrt(self.spk2cmvn[self.utt2spk[utt_name]]['var'])
        elif var == 'norm':
            x_ = x_ / (max_val - min_val)
             
        return x_
