import kaldi_io
import numpy as np
import subprocess
from collections import namedtuple
import torch
import random
import os


Minibatch = namedtuple('Minibatch', ['input', 'target', 'metadata'])


def memmap_feats(feats_scp, f_memmapped, dtype=np.float32):
    '''
        Maps the feats.scp file from kaldi to a memory mapped numpy object.
        This allows for fast i/o when creating window minibatches from slices
        of training data.

        input args: feats_scp, f_memmapped
        output: 
            utt_lens = {'utt_n': # utt_n frames, ... }
            offsets  = {'utt_n': utt_n offset in memory mapped numpy file}
            data_shape = (#frames, feature_dimension)
    '''
    # First get the total lengths of each utterance
    p = subprocess.Popen(
        ['feat-to-len', 'scp:' + feats_scp, 'ark,t:-'],
        stdout=subprocess.PIPE
    )
    out = p.communicate()
    utt_lens = {}
    for l in out[0].split(b'\n'):
        if l.strip() != b'':
            utt_id, utt_len = l.strip().split(None, 1)
            utt_lens[utt_id] = int(utt_len)
    # Next get the dimension of the features
    p = subprocess.Popen(['feat-to-dim', 'scp:' + feats_scp, '-'],
        stdout=subprocess.PIPE
    )
    out = p.communicate()
    dim = int(out[0])
    # Set Data Shape
    data_shape = (sum(utt_lens.values()), dim)
    # Set up memmapped features 
    f = np.memmap(f_memmapped, mode='w+', dtype=dtype, shape=data_shape)
    offsets = {}
    offset = 0
    for i, (k, m) in enumerate(kaldi_io.read_mat_scp(feats_scp)):
        print('Utterance ', i, ' : ', k)
        m = m.astype(dtype)
        offsets[k.encode()] = offset
        new_offset = offset + utt_lens[k.encode()]
        f[offset:new_offset, :] = m
        offset = new_offset
    print()
    del f
    return utt_lens, offsets, data_shape


def get_targets(f_targets):
    '''
        Retrieve the targets (pdfids) corresponding to each input utterance
        input args:
            f_targets -- file pointer to the targets
            
            Format of f_targets:
                utt1 pdf11 pdf12 pdf13 ...
                utt2 pdf21 pdf22 ...
                utt3 ...
                ...
        output:
            utts = {'utt1': [pdf1, pdf2, ...], 'utt2': [pdf1, pdf1, ...]}
    '''
    utts = {}
    for l in f_targets:
        utt_id, tgts = l.strip().split(None, 1)
        if utt_id not in utts:
            utts[utt_id.encode()] = []
        for t in tgts.split():
            utts[utt_id.encode()].append(int(t))
    return utts


def batches(n, dataset):
    '''
        A generator to produce n minibatches from the dataset obejct
        (see dataset.py for definition). Random frames are constructed drawn
        from any position in the memory mapped data.

        Input args:
            n -- number of minibatches
            dataset -- dataset object containing data to be batched
            batchsize -- the number of examples in a minibatch
    '''
    batchsize = dataset.batchsize
    batchlength = dataset.left_context + dataset.right_context + dataset.chunk_width
    batchdim = dataset.data_shape[0][1]
    for b in range(n):
        # Initialize batch
        input_tensor = torch.zeros((batchsize, batchlength, batchdim))
        output = torch.zeros(
            (batchsize, dataset.subsample_chunk_width),
            dtype=torch.int64
        )
        name, index, offset, length = [], [], [], []
        i = 0
        while i < batchsize:
            # first sample a data split at random
            split_idx = random.randint(0, len(dataset.data_shape) - 1)
            # now sample a data point from this split
            # random.randint includes the endpoints
            idx = random.randint(0, dataset.data_shape[split_idx][0] - 1)
            sample = dataset[(split_idx,idx)]

            # Sample did not have paired target
            if sample is None:
                continue;
            
            metadata = sample.metadata 
            # Sample was in the heldout set
            if dataset.utt2spk[metadata['name']] in dataset.heldout:
                continue;

            # Check that the chunk width does not go over the utterance
            # boundary
            start_idx = metadata['index'] - metadata['offset'] 
            if start_idx + dataset.chunk_width > metadata['length']:
                continue;  
            
            name.append(metadata['name'])
            index.append(metadata['index'])
            offset.append(metadata['offset'])
            length.append(metadata['length'])
            input_tensor[i, :, :] = perturb(
                torch.from_numpy(sample.input),
                perturb_type=dataset.perturb_type
            )
            output[i, :] = torch.LongTensor(sample.target)

            i += 1
        metadata = {
            'name': name,
            'index': index,
            'offset': offset,
            'length': length,
            'left_context': dataset.left_context,
            'right_context': dataset.right_context,
        }
        output_tensor = torch.LongTensor(output)
        yield Minibatch(input_tensor, output_tensor, metadata) 
        
        # Memmap will consume more and more RAM if permitted. This periodically
        # forces the buffer to clear by deleting and recreating the memmap.
        dataset.free_ram(split_idx)

def multiset_batches(n, sets):
    '''
        Alternating round-robin batches.
    '''
    set_batches = []
    for s in sets:
        set_batches.append(batches(n, s))

    for i in range(n):
        for set_batch in set_batches:
            b = next(set_batch, None)
            if b is not None:
                yield b
 

def validation_batches(dataset):
    '''
        The utterances correpsonding to heldout speakers are batched in order
        for decoding. This is used for looping through the validation data
        during training.

        Input args:
            dataset -- dataset object (see dataset.py)
            batchsize -- number of examples in a minibatch
    '''
    batchsize = dataset.batchsize
    # Get utterances for each speaker and batch in order
    for s in sorted(dataset.heldout):
        for u in dataset.spk2utt[s]:
            if u in dataset.targets:
                start = dataset.offsets_dict[u]
                end = start + dataset.utt_lengths[u] 
                i = 0
                inputs, output = [], []
                length, offset, index, name = [], [], [], []
                for idx in range(start, end, dataset.chunk_width):
                    sample = dataset[idx]
                    metadata = sample.metadata
                    name.append(metadata['name'])
                    index.append(metadata['index'])
                    offset.append(metadata['offset'])
                    length.append(metadata['length'])
                    inputs.append(sample.input) 
                    output.extend(sample.target)
                    i += 1
                    # Yield the minibatch when this one is full 
                    if i == batchsize:
                        metadata = {
                            'name': name, 'index': index,
                            'offset': offset, 'length': length,
                            'left_context': dataset.left_context,
                            'right_context': dataset.right_context,
                        }
                        input_tensor = torch.tensor(inputs, dtype=torch.float32) 
                        output_tensor = torch.LongTensor(output)
                        yield Minibatch(input_tensor, output_tensor, metadata) 
                        i = 0
                        inputs, output = [], []
                        length, offset, index, name = [], [], [], []
                # Yield the minibatch when the utterance is done
                if i > 0:
                    metadata = {
                        'name': name, 'index': index,
                        'offset': offset, 'length': length,
                        'left_context': dataset.left_context,
                        'right_context': dataset.right_context,
                    }
                    input_tensor = torch.tensor(inputs, dtype=torch.float32) 
                    output_tensor = torch.LongTensor(output)
                    yield Minibatch(input_tensor, output_tensor, metadata) 


def evaluation_batches(dataset):
    '''
        Identical to validation_batches, but selecting only those utterances
        in the utt_subset field of dataset.
    '''
    batchsize = dataset.batchsize
    # Get utterances for each speaker and batch in order
    for u in dataset.utt_subset:
        start = dataset.offsets_dict[u]
        end = start + dataset.utt_lengths[u] 
        i = 0
        inputs, output = [], []
        length, offset, index, name = [], [], [], []
        for idx in range(start, end, dataset.chunk_width):
            sample = dataset[idx]
            metadata = sample.metadata
            name.append(metadata['name'])
            index.append(metadata['index'])
            offset.append(metadata['offset'])
            length.append(metadata['length'])
            inputs.append(sample.input) 
            output.extend(sample.target)
            i += 1
            # Yield the minibatch when this one is full 
            if i == batchsize:
                metadata = {
                    'name': name, 'index': index,
                    'offset': offset, 'length': length,
                    'left_context': dataset.left_context,
                    'right_context': dataset.right_context,
                }
                input_tensor = torch.tensor(inputs, dtype=torch.float32) 
                output_tensor = torch.LongTensor(output)
                yield Minibatch(input_tensor, output_tensor, metadata) 
                i = 0
                inputs, output = [], []
                length, offset, index, name = [], [], [], []
        # Yield the minibatch when the utterance is done
        if i > 0:
            metadata = {
                'name': name, 'index': index,
                'offset': offset, 'length': length,
                'left_context': dataset.left_context,
                'right_context': dataset.right_context,
            }
            input_tensor = torch.tensor(inputs, dtype=torch.float32) 
            output_tensor = torch.LongTensor(output)
            yield Minibatch(input_tensor, output_tensor, metadata) 


def move_to(b, device):
    '''
        Move minibatch b to device
    '''
    return Minibatch(b.input.to(device), b.target.to(device), b.metadata)


def load_cmvn(filename):
    '''
        Load the cmvn file. Requires filename.
    '''
    gen = kaldi_io.read_mat_scp(filename)
    spk2cmvn = {}
    for k, m in gen:
        total = m[0, -1]
        spk2cmvn[k] = {'mu': m[0, :-1] / total, 'var': m[1, :-1] / total}
    return spk2cmvn


def load_utt2spk(f):
    '''
        Load the utt2spk file. Requires an open file pointer.
    '''
    utt2spk = {}
    for l in f:
        utt, spk = l.strip().split(None, 1)
        utt2spk[utt.encode()] = spk
    spk2utt = {}
    for u, s in utt2spk.items():
        if s not in spk2utt:
            spk2utt[s] = []
        spk2utt[s].append(u)
    return utt2spk, spk2utt


def load_segments(f):
    '''
        Load the segments file. Requires an open file pointer.
    '''
    audio_to_segments = {}
    for l in f:
        utt, audio, start, end = l.strip().split()
        if audio not in audio_to_segments:
            audio_to_segments[audio] = []
        audio_to_segments[audio].append((utt, start, end))
    return audio_to_segments


def load_utt_subset(f):
    '''
        Load the subset of utterances from file point f. Use a kaldi segments
        file for the file f for example.
    '''
    utt_subset = []
    for l in f:
        utt_subset.append(l.strip().split(None, 1)[0].encode())
    return utt_subset


def perturb(x, perturb_type='none'):
    if perturb_type == 'none':
        return x
    elif perturb_type == 'salt_pepper':
        x *= torch.FloatTensor(x.size()).random_(0, 2).to(x.dtype)
    elif perturb_type == 'time_mask':
        width=20
        start = random.randint(0, x.size(1) - width)
        end = start + width
        mask = (torch.arange(x.size(1)) >= start) * (torch.arange(x.size(1)) < end)  
        mask = mask[:, None].expand(x.size())
        x[mask] = 0.0
    elif perturb_type == 'freq_mask': 
        width=10
        start = random.randint(0, x.size(0) - width)
        end = start + width
        mask = (torch.arange(x.size(-1)) >= start) * (torch.arange(x.size(-1)) < end)  
        mask = mask[None, :].expand(x.size())
        x[mask] = 0.0 
    return x 
