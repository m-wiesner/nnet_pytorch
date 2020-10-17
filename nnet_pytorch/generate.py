#! /usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2020 
# Apache 2.0

import os
import argparse
import json
import subprocess
import numpy as np
import torch
import models
import objectives
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
    
    cw = args.chunk_width
    cw += args.left_context + args.right_context
    
    samples = objective.generate_from_model(
        model,
        bs=args.batchsize,
        cw=cw,
        dim=args.idim,
        left_context=args.left_context, right_context=args.right_context,
        device=device
    )

    for i, s in enumerate(samples):
        np.save('{}/samples.{}'.format(args.dumpdir, i), s)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='model directory used for generated')
    parser.add_argument('--dumpdir', help='dump results here')
    parser.add_argument('--modelname', default='final.mdl')
    parser.add_argument('--gpu', action='store_true', help='Tun on gpu. This '
        'can be very slow on cpu'
    )
    parser.add_argument('--idim', type=int, default=64,
        help='The input dimension of features'
    )
    parser.add_argument('--chunk-width', type=int, default=50,
        help='The width of the speech chunk. The target sequence will be '
        'length chunk_width / subsample'
    )
    parser.add_argument('--left-context', type=int, default=10,
        help='extra left context on the input features'
    )
    parser.add_argument('--right-context', type=int, default=5,
        help='extra right context on the input features'
    )
    parser.add_argument('--batchsize', type=int, default=32,
        help='number of sample to generate (just 1 minibatch)',
    )
   
    # Args specific to different components
    args, leftover = parser.parse_known_args()
    conf = json.load(open(args.modeldir + '/conf.1.json'))
    models.MODELS[conf['model']].add_args(parser) 
    parser.parse_args(leftover, namespace=args) 
    return args
  
   
if __name__ == "__main__":
    main()

