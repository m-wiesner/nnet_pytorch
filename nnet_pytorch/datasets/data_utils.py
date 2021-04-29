import kaldi_io
import numpy as np
import subprocess
import torch
import random


def memmap_feats(feats_scp, f_memmapped, utt_list, dtype=np.float32):
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
        if k not in utt_list:
            continue;
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


def load_utt2num_frames(f):
    '''
        Load the utt2num_frames files produced during feature creation.
    '''
    utt2num_frames = {}
    for l in f:
        utt, num_frames = l.strip().split(None, 1)
        utt2num_frames[utt] = int(num_frames)
    return utt2num_frames


def perturb(x, perturbations='none'):
    def apply_perturbation(x, perturb_type): 
        if perturb_type[0] == 'salt_pepper':
            maxval = perturb_type.split()[1]
            x *= torch.FloatTensor(x.size()).random_(0, maxval).to(x.dtype)
        elif perturb_type[0] == 'time_mask':
            params = perturb_type[1]   
            width = params.get('width', 4) # 4 is the default
            max_drop_percent = params.get('max_drop_percent', None)
            if max_drop_percent is None: 
                num_holes = params.get('holes', 2) # 2 is the default
            else:
                # Max number of holes to create
                num_holes = int(max_drop_percent * (x.size(0) / width))  
            for i in range(random.randint(0, num_holes)):
                # The width has to be at most 1 less than x.size(0)
                # where x.size(0) = left + cw + right
                this_width = min(int(width * random.random()), x.size(0) - 1)
                start = random.randint(0, x.size(0) - this_width)
                end = start + this_width
                mask = (torch.arange(x.size(0)) >= start) * (torch.arange(x.size(0)) < end)  
                mask = mask[:, None].expand(x.size())
                x[mask] = 0.0
        elif perturb_type[0] == 'freq_mask': 
            params = perturb_type[1]
            width = params.get('width', 4) # 4 is the default
            num_holes = params.get('holes', 2) # 2 is the default
            for i in range(random.randint(0, num_holes)):
                # The width has to be at most 1 less than x.size(-1) 
                this_width = min(int(width * random.random()), x.size(-1) - 1)
                start = random.randint(0, x.size(-1) - this_width)
                end = start + this_width
                mask = (torch.arange(x.size(-1)) >= start) * (torch.arange(x.size(-1)) < end)  
                mask = mask[None, :].expand(x.size())
                x[mask] = 0.0 
        elif perturb_type[0] == 'gauss':
            params = perturb_type[1]
            std = params.get('std', 0.3) # 0.3 is the default   
            this_std = std * random.random()
            x += this_std * torch.randn_like(x)
        elif perturb_type[0] == 'rand':
            params = perturb_type[1]
            maxval = params.get('maxval', 1.0) 
            x.uniform_(-maxval, maxval)
        elif perturb_type[0] == 'volume':
            params = perturb_type[1]
            maxscale = params.get('scale', 2.0)
            this_scale = maxscale * random.random()
            x *= this_scale

    
    if perturbations == 'none':
        return
    else:
        perturbations = eval(perturbations)
    for pt in perturbations:
        apply_perturbation(x, pt)
