import numpy as np
import torch
from bisect import bisect
from collections import namedtuple
from .data_utils import *
from .NnetPytorchDataset import NnetPytorchDataset
import kaldi_io
import os
import pickle
import random
from itertools import groupby


class HybridAsrDataset(NnetPytorchDataset):

    Minibatch = namedtuple('Minibatch', ['input', 'target', 'metadata'])

    @staticmethod
    def add_args(parser):
        parser.add_argument('--perturb-type', type=str, default='none')
        parser.add_argument('--utt-subset', default=None)

    @classmethod
    def build_dataset(cls, ds):
        return HybridAsrDataset(
            ds['data'], ds['tgt'],
            left_context=ds['left_context'],
            right_context=ds['right_context'],
            chunk_width=ds['chunk_width'],
            batchsize=ds['batchsize'],
            subsample=ds['subsample'],
            mean=ds['mean_norm'], var=['var_norm'],
            utt_subset=ds['utt_subset']
        )
    
    def __init__(self, datadir, targets,
        dtype=np.float32, memmap_affix='.dat',
        left_context=10, right_context=3, chunk_width=1,
        batchsize=128, mean=True, var='norm',
        utt_subset=None, subsample=1,
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
       
        # num splits
        with open(os.path.sep.join((datadir, 'num_split'))) as f:
            num_split = int(f.readline().strip())

                
        # Dump memmapped features (Faster I/O and no egs creation)
        feats_scp = os.path.sep.join((datadir, 'feats.scp'))
        f_memmap = feats_scp + memmap_affix
        metadata_path = os.path.sep.join((datadir, 'mapped', 'metadata'))
        self.data_path = os.path.sep.join((datadir, 'mapped', 'feats.dat'))
        utt_lengths = {}
        offsets = []
        data_shape = []
        for n in range(1, 1 + num_split):
            with open(metadata_path + '.' + str(n) + '.pkl', 'rb') as f:
                utt_lengths_n, offsets_n, data_shape_n = pickle.load(f)
                utt_lengths.update(utt_lengths_n)
                offsets.append(offsets_n)
                data_shape.append(data_shape_n)
       
        # Creates 1 file pointer per memmap split   
        print(dtype)
        self.data = [
            np.memmap(
                "{}.{}".format(self.data_path, n),
                dtype=dtype,
                shape=data_shape[n-1],
                mode='r',
            ) for n in range(1, 1 + num_split)
        ]

        # Get some metadata
        self.utt_lengths = utt_lengths # list of dicts with (utt, len) (key, val) pairs
        
        # dict with (utt, offset) (key, val) pairs. This is needed for inference
        self.offsets_dict = {
            u: (split_idx, v) for split_idx, offsets_n in enumerate(offsets)
                for u, v in offsets_n.items()
        }
        
        self.utt_offsets = [sorted(offset_n.items(), key=lambda x: x[1]) for offset_n in offsets] # sorted list of utterances by offset
        self.offsets = [[u[1] for u in offset_n] for offset_n in self.utt_offsets] # list of sorted list of offsets
        self.data_shape = data_shape # list of shape of the whole data (per split)

        self.split_offsets = [0]
        for shape in self.data_shape:
            self.split_offsets.append(self.split_offsets[-1] + shape[0])
        
        # Get targets. Dummy targets for utterances where labels are unknown
        with open(targets) as f:
            self.targets = get_targets(f)
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
        '''
            This function flushes the memmap memory buffer for a particular 
        '''
        dtype = self.data[0].dtype
        del self.data[split_idx]
        # file indexes are 1-based while split_idx is 0-based
        self.data.insert(
            split_idx, np.memmap(
                "{}.{}".format(self.data_path, split_idx + 1),
                dtype=dtype,
                shape=self.data_shape[split_idx],
                mode='r',
            )
        )  

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
        # Find which utterance the index belongs to
        split_idx, utt_idx, idx = index
        utt_name, offset = self.utt_offsets[split_idx][utt_idx]
        utt_length = self.utt_lengths[utt_name]
                
        # Retrieve the appropriate target
        target_start = (idx - offset) // self.subsample
        target_end = min(
            len(self.targets[utt_name]),
            target_start + self.subsample_chunk_width,
        )
                
        target = self.targets[utt_name][target_start: target_end]
        
        # Get the lower and upper boundaries for the window
        lower_boundary = max(offset, idx - self.left_context)
        upper_boundary = min(
            offset + utt_length, idx + self.right_context + self.chunk_width
        )

        x = np.array(self.data[split_idx][lower_boundary: upper_boundary, :])
                
        # Apply cmvn
        if self.mean or (self.var != 'none'):
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
        }
        return HybridAsrDataset.Minibatch(x, target, metadata)

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

    def __len__(self):
        return sum(s[0] for s in self.data_shape)
    
    def size(self, idx):
        return 1

    def minibatch(self):
        batchlength = self.left_context + self.right_context + self.chunk_width
        batchdim = self.data_shape[0][1]
        # Initialize batch
        input_tensor = torch.zeros((self.batchsize, batchlength, batchdim))
        output = torch.zeros(
            (self.batchsize, self.subsample_chunk_width),
            dtype=torch.int64
        )
        name, split = [], []
        size = 0
        while size < self.batchsize:
            # first sample a data split at random
            split_idx = int((random.random() - 1e-09) * len(self.data_shape))
            
            # now sample a data point from this split
            # random.randint includes the endpoints
            utt_idx = int((random.random() - 1e-09) * len(self.utt_offsets[split_idx]))
            utt_name, utt_offset = self.utt_offsets[split_idx][utt_idx]
            utt_length = self.utt_lengths[utt_name]
            idx = int((random.random() - 1e-09) * utt_length)

            # Check that the chunk width does not go over the utterance
            # boundary
            if idx + self.chunk_width > utt_length:
                continue; 
            
            split.append(split_idx)
            sample = self[(split_idx, utt_idx, idx + utt_offset)]
            name.append(sample.metadata['name'])
        
            perturb_type = 'none'   
            if sample.target[0] == -1:
                perturb_type = 'gauss'
         
            input_tensor[size, :, :] = perturb(
                torch.from_numpy(sample.input),
                perturb_type=perturb_type,
            )
            output[size, :] = torch.LongTensor(sample.target)

            size += self.size(idx)
        
        metadata = {
            'name': name,
            'left_context': self.left_context,
            'right_context': self.right_context,
        }

        output_tensor = torch.LongTensor(output)
        self.closure(set(split))
        return HybridAsrDataset.Minibatch(input_tensor, output_tensor, metadata) 
        

    def evaluation_batches(self):
        for u in self.utt_subset:
            split_idx, start = self.offsets_dict[u]
            utt_idx = max(0, bisect(self.offsets[split_idx], start) - 1)
            end = start + self.utt_lengths[u] 
            i = 0
            inputs, output = [], []
            name, split = [], []
            for idx in range(start, end, self.chunk_width):
                sample = self[(split_idx, utt_idx, idx)]
                name.append(sample.metadata['name'])
                split.append(split_idx)
                inputs.append(sample.input) 
                output.extend(sample.target)
                i += 1
                # Yield the minibatch when this one is full 
                if i == self.batchsize:
                    metadata = {
                        'name': name, 
                        'left_context': self.left_context,
                        'right_context': self.right_context,
                    }
                    input_tensor = torch.tensor(inputs, dtype=torch.float32) 
                    output_tensor = torch.LongTensor(output)
                    yield HybridAsrDataset.Minibatch(input_tensor, output_tensor, metadata) 
                    i = 0
                    inputs, output = [], []
                    name = []
            # Yield the minibatch when the utterance is done
            if i > 0:
                metadata = {
                    'name': name, 
                    'left_context': self.left_context,
                    'right_context': self.right_context,
                }
                input_tensor = torch.tensor(inputs, dtype=torch.float32) 
                output_tensor = torch.LongTensor(output)
                yield HybridAsrDataset.Minibatch(input_tensor, output_tensor, metadata) 
                self.closure(set(split))

    
    def closure(self, splits):
        for idx in splits:
            self.free_ram(idx)

    
    def move_to(b, device):
        '''
            Move minibatch b to device
        '''
        return HybridAsrDataset.Minibatch(b.input.to(device), b.target.to(device), b.metadata)
