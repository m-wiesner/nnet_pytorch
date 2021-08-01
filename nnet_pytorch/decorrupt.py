#!/usr/bin/env python
# Copyright 2021
# Apache 2.0


import os
import sys
import argparse
import json
import subprocess
import numpy as np
import torch
import datasets
import models
import objectives
from batch_generators import evaluation_batches
from IterationTypes import decorrupt_dataset
import socket


def main():
    args = parse_arguments()
    print(args)

    hostname = socket.gethostname()  
    # Reserve the GPU if used in decoding. In general it won't be.        
    if args.gpu and 'clsp' in hostname:
        # User will need to set CUDA_VISIBLE_DEVICES here
        cvd = subprocess.check_output(["/usr/local/bin/free-gpu", "-n", "1"]).decode().strip()
        os.environ['CUDA_VISIBLE_DEVICES'] = cvd
    
    device = torch.device('cuda' if args.gpu else 'cpu')
    reserve_variable = torch.ones(1).to(device)

    # Load experiment configurations so that decoding uses the same parameters
    # as training
    conf = json.load(open(args.modeldir + '/conf.1.json'))
    
    dataset_args = eval(conf['datasets'])[0]
    
    # Load the decoding dataset
    subsample_val = 1
    if 'subsample' in conf:
        subsample_val=conf['subsample']
   
    # Note here that the targets file is just a dummy placeholder. We don't
    # need the targets or use them. The keyword argument validation=0, is
    # because we are decoding and do not have the targets so validation does
    # not make sense in this context.
    targets = args.targets 
    if not os.path.exists(targets):
        raise IOError(f"{targets} not found.")
    
    dataset_args.update(
        {
            'data':args.datadir,
            'tgt':targets,
            'subsample': subsample_val,
            'utt_subset': args.utt_subset,
            'perturb_type': args.perturb,
            'chunk_width': args.chunk_width,
            'left_context': args.left_context,
            'right_context': args.right_context,
        }
    )
   
    dataset = datasets.DATASETS[conf['datasetname']].build_dataset(dataset_args)
    print(conf)

    objective = objectives.OBJECTIVES[conf['objective']].build_objective(conf)
    objective.to(device)
    # Build the model and send to the device (cpu or gpu). Generally cpu.
    model = models.MODELS[conf['model']].build_model(conf)
    model.to(device)
  
    # Load the model from experiment checkpoint 
    mdl = torch.load(
        os.path.sep.join([args.modeldir, args.checkpoint]),
        map_location=device
    )
    model.load_state_dict(mdl['model'])
    objective.load_state_dict(mdl['objective'])
    
    args.datasetname = conf['datasetname']
    decorrupt(args, dataset, model, objective, device=device)


def decorrupt(args, dataset, model, objective, device='cpu'):
    '''
        Produce lattices from the input utterances.
    '''
    model.eval()
    utt_mats = {} 
    prev_key = b''
    stride = args.left_context + args.chunk_width + args.right_context
    delay = args.left_context
    generator = evaluation_batches(dataset, stride=stride, delay=delay)
    # Each minibatch is guaranteed to have at most 1 utterance. We need
    # to append the output of subsequent minibatches corresponding to
    # the same utterances. These are stored in ``utt_mat'', which is
    # just a buffer to accumulate the posterior outputs of minibatches
    # corresponding to the same utterance. The posterior state
    # probabilities are normalized (subtraction in log space), by the
    # log priors in order to produce pseudo-likelihoods useable for
    # for lattice generation with latgen-faster-mapped
    for i, (key, sgld_iter, mat, targets) in enumerate(decorrupt_dataset(args, generator, model, objective, device=device)):
        print(f"key: {key} sgld_iter: {sgld_iter}")
        print(f"targets: {targets}")
        if sgld_iter not in utt_mats:
            utt_mats[sgld_iter] = []

        if len(utt_mats[sgld_iter]) > 0 and key != prev_key:   
            utt_length = dataset.utt_lengths[prev_key] 
            for sgld_iter_ in utt_mats:
                np.save(
                    '{}/{}.{}'.format(args.dumpdir, prev_key.decode('utf-8'), str(sgld_iter_)),
                    np.concatenate(utt_mats[sgld_iter_], axis=0)[:utt_length, :],
                )
            utt_mats = {sgld_iter: []}

        utt_mats[sgld_iter].append(mat)
        prev_key = key

    # Flush utt_mat buffer at the end
    if len(utt_mats) > 0:
        utt_length = dataset.utt_lengths[prev_key] 
        if len(utt_mats[0]) > 0:
            for sgld_iter in utt_mats:
                np.save(
                    '{}/{}.{}'.format(args.dumpdir, prev_key.decode('utf-8'), str(sgld_iter)),
                    np.concatenate(utt_mats[sgld_iter], axis=0)[:utt_length, :],
                )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir')
    parser.add_argument('--modeldir')
    parser.add_argument('--dumpdir')
    parser.add_argument('--targets', type=str, default=None)
    parser.add_argument('--checkpoint', default='final.mdl')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--idim', type=int, default=64)
    parser.add_argument('--chunk-width', type=int, default=50)
    parser.add_argument('--left-context', type=int, default=10)
    parser.add_argument('--right-context', type=int, default=5)
    parser.add_argument('--num-steps', type=int, default=None)
    parser.add_argument('--perturb', type=str, default='none')
   
    # Args specific to different components
    args, leftover = parser.parse_known_args()
    conf = json.load(open(args.modeldir + '/conf.1.json'))
    datasets.DATASETS[conf['datasetname']].add_args(parser)
    models.MODELS[conf['model']].add_args(parser) 
    parser.parse_args(leftover, namespace=args) 
    return args


if __name__ == "__main__":
    main() 
