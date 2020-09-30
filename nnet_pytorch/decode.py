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
    mean_norm, var_norm = eval(conf['mean_var'])
    
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
    
    dataset = datasets.DATASETS[conf['datasetname']](
        args.datadir, targets, conf['num_targets'],
        left_context=dataset_args['left_context'],
        right_context=dataset_args['right_context'],
        chunk_width=dataset_args['chunk_width'],
        batchsize=args.batchsize,
        validation=0, utt_subset=args.utt_subset,
        subsample=subsample_val,
        mean=mean_norm, var=var_norm,
    )

    # We just need to add in the input dimensions. This depends on the type of
    # features used.
    conf['idim'] = dataset.data_shape[0][1] 
    
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
   
    # Load the state priors (For x-ent only systems)
    priors = 0.0
    if 'LFMMI' not in conf['objective']:
        priors = json.load(open(os.path.join(args.modeldir, args.prior_name)))
        priors = np.array(priors)
    
        # Floor likelihoods (by altering the prior) for states with very low
        # priors
        priors[priors < args.prior_floor] = 1e20
    args.objective = conf['objective']
    decode(args, dataset, model, priors, device=device)


def decode(args, dataset, model, priors, device='cpu'):
    '''
        Produce lattices from the input utterances.
    '''
    # This is all of the kaldi code we are calling. We are just piping out
    # out features to latgen-faster-mapped which does all of the lattice
    # generation.
    lat_output = '''ark:| copy-feats ark:- ark:- |\
    latgen-faster-mapped --min-active={} --max-active={} \
    --max-mem={} \
    --lattice-beam={} --beam={} \
    --acoustic-scale={} --allow-partial=true \
    --word-symbol-table={} \
    {} {} ark:- ark:- | lattice-scale --acoustic-scale={} ark:- ark:- |\
    gzip -c > {}/lat.{}.gz'''.format(
        args.min_active, args.max_active, args.max_mem,
        args.lattice_beam, args.beam, args.acoustic_scale,
        args.words_file, args.trans_mdl, args.hclg,
        args.post_decode_acwt, args.dumpdir, args.job
    )
    
    # Do the decoding (dumping senone posteriors)
    model.eval()
    with torch.no_grad():
        with kaldi_io.open_or_fd(lat_output, 'wb') as f:
            utt_mat = [] 
            prev_key = b''
            generator = evaluation_batches(dataset)
            # Each minibatch is guaranteed to have at most 1 utterance. We need
            # to append the output of subsequent minibatches corresponding to
            # the same utterances. These are stored in ``utt_mat'', which is
            # just a buffer to accumulate the posterior outputs of minibatches
            # corresponding to the same utterance. The posterior state
            # probabilities are normalized (subtraction in log space), by the
            # log priors in order to produce pseudo-likelihoods useable for
            # for lattice generation with latgen-faster-mapped
            for key, mat in decode_dataset(args, generator, model, device='cpu'):
                if len(utt_mat) > 0 and key != prev_key:   
                    kaldi_io.write_mat(
                        f, np.concatenate(utt_mat, axis=0)[:utt_length, :],
                        key=prev_key.decode('utf-8')
                    )
                    utt_mat = []
                utt_mat.append(mat - args.prior_scale * priors)
                prev_key = key
                utt_length = dataset.utt_lengths[key] // dataset.subsample 

            # Flush utt_mat buffer at the end
            if len(utt_mat) > 0:
                kaldi_io.write_mat(
                    f,
                    np.concatenate(utt_mat, axis=0)[:utt_length, :],
                    key=prev_key.decode('utf-8')
                )

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir')
    parser.add_argument('--modeldir')
    parser.add_argument('--dumpdir')
    parser.add_argument('--checkpoint', default='final.mdl')
    parser.add_argument('--prior-scale', type=float, default=1.0)
    parser.add_argument('--prior-floor', type=float, default=-20)
    parser.add_argument('--prior-name', default='priors')
    parser.add_argument('--words-file')
    parser.add_argument('--trans-mdl')
    parser.add_argument('--hclg')
    parser.add_argument('--min-active', type=int, default=200)
    parser.add_argument('--max-active', type=int, default=7000)
    parser.add_argument('--max-mem', type=int, default=50000000)
    parser.add_argument('--lattice-beam', type=float, default=8.0)
    parser.add_argument('--beam', type=float, default=15.0)
    parser.add_argument('--acoustic-scale', type=float, default=0.1)
    parser.add_argument('--post-decode-acwt', type=float, default=1.0)
    parser.add_argument('--job', type=int, default=1)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--batchsize', type=int, default=256)
   
    # Args specific to different components
    args, leftover = parser.parse_known_args()
    conf = json.load(open(args.modeldir + '/conf.1.json'))
    HybridAsrDataset.add_args(parser)
    models.MODELS[conf['model']].add_args(parser) 
    parser.parse_args(leftover, namespace=args) 
    return args


if __name__ == "__main__":
    main() 
