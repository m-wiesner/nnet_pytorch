#!/usr/bin/env python

import os
import sys
import argparse
import json
import subprocess
import numpy as np
import torch
import datasets
import models
from LRScheduler import LRScheduler
from batch_generators import evaluation_batches
from IterationTypes import decode_dataset
import kaldi_io


def main():
    args = parse_arguments()
    print(args)

    # Reserve the GPU if used in decoding. In general it won't be.        
    if args.gpu:
        # User will need to set CUDA_VISIBLE_DEVICES here
        cvd = subprocess.check_output(["/usr/bin/free-gpu", "-n", "1"]).decode().strip()
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
    targets = os.path.join(args.datadir, 'pdfid.{}.tgt'.format(str(subsample_val)))
    if not os.path.exists(targets):
        print("Dummy targets not found")
        sys.exit(1)
    
    dataset_args.update(
        {
            'data':args.datadir,
            'tgt':targets,
            'subsample': subsample_val,
            'utt_subset': args.utt_subset,
        }
    )
    
    dataset = datasets.DATASETS[conf['datasetname']].build_dataset(dataset_args)
    print(conf) 
    # Build the model and send to the device (cpu or gpu). Generally cpu.
    model = models.MODELS[conf['model']].build_model(conf)
    model.to(device)
  
    # Load the model from experiment checkpoint 
    mdl = torch.load(
        os.path.sep.join([args.modeldir, args.checkpoint]),
        map_location=device
    )
    model.load_state_dict(mdl['model'])
   
    args.objective = conf['objective']
    args.datasetname = conf['datasetname']
    forward(args, dataset, model, device=device)


def forward(args, dataset, model, device='cpu'):
    model.eval()
    with torch.no_grad():
        generator = batches(args.num_samples, dataset)
        for i, v in enumerate(evaluate_energies(args, generator, model, device=device))
            np.save('{}/embeddings.{}'.format(args.dumpdir, str(i)), v.cpu().numpy())


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir')
    parser.add_argument('--modeldir')
    parser.add_argument('--dumpdir')
    parser.add_argument('--checkpoint', default='final.mdl')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--num-samples', type=int, default=1000)
   
    # Args specific to different components
    args, leftover = parser.parse_known_args()
    conf = json.load(open(args.modeldir + '/conf.1.json'))
    datasets.DATASETS[conf['datasetname']].add_args(parser)
    models.MODELS[conf['model']].add_args(parser) 
    parser.parse_args(leftover, namespace=args) 
    return args


if __name__ == "__main__":
    main() 
