#! /usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2020 
# Apache 2.0

import os
import sys
import argparse
import json
import random
import subprocess
import numpy as np
import torch
import models
import kaldi_io
import objectives
from functools import partial
from collections import namedtuple
from data_utils import move_to


Samples = namedtuple('Samples', ['input', 'target', 'metadata']) 


def main():
    args = parse_arguments()
    print(args)
  
    # Reserve the GPU if used in decoding. 
    if args.gpu: 
        # USER will need to set CUDA_VISIBLE_DEVICES here
        cvd = subprocess.check_output(["/usr/local/bin/free-gpu", "-n", "1"]).decode().strip()
        os.environ['CUDA_VISIBLE_DEVICES'] = cvd
    
    device = torch.device('cuda' if args.gpu else 'cpu')
    reserve_variable = torch.ones(1).to(device)
   
    # Load experiment configurations so that decoding uses the same parameters
    # as training
    conf = json.load(open(args.modeldir + '/conf.1.json'))
    conf['idim'] = args.idim
    if args.chunk_width:
        conf['chunk_width'] = args.chunk_width
    # Build the model and send to the device (cpu or gpu). Generally cpu.
    objective = objectives.OBJECTIVES[conf['objective']].build_objective(conf)
    objective.to(device)
    model = models.MODELS[conf['model']].build_model(conf)
    model.to(device)

    mdl = torch.load(
        os.path.sep.join([args.modeldir, args.modelname]),
        map_location=device
    )
    objective.load_state_dict(mdl['objective']) 
    model.load_state_dict(mdl['model'])  
    buff = objective.generate_from_buffer()
   
    # Get the chunkwidth
    cw = args.chunk_width
    cw += args.left_context + args.right_context
    
    metadata = {
        'left_context': args.left_context,
        'right_context': args.right_context,
    }
    obj = objectives.OBJECTIVES['LFMMINum']
    if args.target is not None:
        target = torch.LongTensor(args.batchsize*[args.target])
    else:
        target = -1
        for i in range(args.top_k):
            idx = random.randint(0, buff.size(0) - 1)
            np.save(args.dumpdir + '/example_' + str(i), buff[idx].data.cpu().numpy())
        sys.exit()

    # Start the sampling
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
   
    # We score all of the generated samples in the training buffer. These are
    # by definition generated samples of the model. We score the samples
    # according to a specified output target sequence. The sequence could be
    # as long as T/subsample where T is the length of samples in the buffer.
    buff_scores = []
    for i in range(0, len(buff), args.batchsize):
        print("Iter: ", i)
        x = buff[i:i+args.batchsize]
        sample = move_to(Samples(x, target, metadata), device)
        model_output = model(sample)
        targets = sample.target
        acoustic_costs = [
            sum(
                [
                model_output[0][i, t, targets[i, t]]
                        for t in range(model_output[0].size(1))
                ]
            ).data.item() for i in range(model_output[0].size(0))
        ]
        buff_scores.extend(acoustic_costs)
    buff_idx = [i for _, i in sorted(zip(buff_scores, range(len(buff))), reverse=True)]
    buff_good = buff[buff_idx[0:args.top_k]]
    buff_bad = buff[buff_idx[-args.top_k:]]
    np.save(
        args.dumpdir + '/pos_tgt_' + '_'.join([str(i) for i in args.target]),
        buff_good.cpu().data.numpy()
    )
    np.save(
        args.dumpdir + '/neg_tgt_' + '_'.join([str(i) for i in args.target]),
        buff_bad.cpu().data.numpy()
    )



def load_source(args, cw):
    if args.source is not None:
        utts_dict = kaldi_io.read_mat_scp(args.source)
        i = 0
        mats = []
        while i < args.num_segments:
            key, mat = next(utts_dict, None)
            if ((mat.shape[0] - cw) < 0):
                start_frame = random.randint(0, mat.shape[0] - cw)
                mats.append(mat)
                i += 1
        x_source = torch.from_numpy(np.array(mats).astype(np.float32))
        x = perturb(x_source, args)
    else:
        x = torch.FloatTensor(args.num_segments, cw, args.idim).uniform_(-1, 1)
    return x


def perturb(x, args):
    if args.perturbation is 'salt_pepper':
        x *= torch.FloatTensor(x.size()).random_(0, 2)
    elif args.perturbation is 'time_mask':
        width=20
        start = random.randint(0, x.size(1) - width)
        end = start + width
        mask = (torch.arange(x.size(1)) >= start) * (torch.arange(x.size(1)) < end)  
        mask = mask[None, :, None].expand(x.size())
        x[mask] = 0.0
    elif args.perturbationis is 'freq_mask': 
        width=10
        start = random.randint(0, x.size(0) - width)
        end = start + width
        mask = (torch.arange(x.size(2)) >= start) * (torch.arange(x.size(2)) < end)  
        mask = mask[None, None, :].expand(x.size())
        x[mask] = 0.0 
    else:
        return x 


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', nargs='+', type=int,
        help='The target sequence against which samples are compared'
    )
    parser.add_argument('--idim', type=int, default=80,
        help='The input dimension of features'
    )
    parser.add_argument('--chunk-width', type=int, default=10,
        help='The width of the speech chunk. The target sequence will be '
        'length chunk_width / subsample'
    )
    parser.add_argument('--left-context', type=int, default=40,
        help='extra left context on the input features'
    )
    parser.add_argument('--right-context', type=int, default=10,
        help='extra right context on the input features'
    )
    parser.add_argument('--modeldir', help='model directory used for generated')
    parser.add_argument('--dumpdir', help='dump results here')
    parser.add_argument('--modelname', default='final.mdl')
    parser.add_argument('--batchsize', default=1, type=int,
        help='batchsize to use in the forward pass when decoding'
    )
    parser.add_argument('--top-k', default=10, type=int, 
        help='The number of best matches from the buffer to output'
    )
    parser.add_argument('--gpu', action='store_true', help='Tun on gpu. This '
        'can be very slow on cpu'
    )
   
    # Args specific to different components
    args, leftover = parser.parse_known_args()
    conf = json.load(open(args.modeldir + '/conf.1.json'))
    models.MODELS[conf['model']].add_args(parser) 
    parser.parse_args(leftover, namespace=args) 
    return args
  
   
if __name__ == "__main__":
    main()

