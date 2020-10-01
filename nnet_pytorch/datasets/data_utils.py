import kaldi_io
import numpy as np
import subprocess
import torch
import random


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

def load_ivectors(filename):
    '''
        Load the ivectors into a dictionary.
        Input argument may be an ark or scp file.
    '''
    ivectors = {}
    for key, vec in kaldi_io.read_vec_flt_scp(filename):
        ivectors[key] = np.array(vec)
    return ivectors


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
