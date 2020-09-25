#! /usr/bin/env python

import os
import argparse
import json
import subprocess
import random
import numpy as np
import torch
from dataset import HybridAsrDataset
import models
import objectives
from LRScheduler import LRScheduler
from itertools import chain
from data_utils import multiset_batches, batches, validation_batches
from IterationTypes import train_epoch, validate, decode_dataset


def main():
    args = parse_arguments() 
    print(args)
  
    # Get GPU and reserve
    if args.gpu:
        # User will need to set CUDA_VISIBLE_DEVICES here
        cvd = subprocess.check_output(["/usr/local/bin/free-gpu", "-n", "1"]).decode().strip()
        os.environ['CUDA_VISIBLE_DEVICES'] = cvd

    device = torch.device('cuda' if args.gpu else 'cpu')
    reserve_variable = torch.ones(1).to(device)

    # Set the random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Resume training / dump training configurations
    if args.resume is not None:
        print("Loading former training configurations ...")
        conf = json.load(
            open('{}/conf.{}.json'.format(args.expdir, args.job))
        )
        if conf['objective'] in ('SemisupLFMMI', 'LFMMI_EBM'):
            conf['l2_energy'] = args.l2_energy 
    else:
        # Dump training configurations
        json.dump(
            vars(args),
            open('{}/conf.{}.json'.format(args.expdir, args.job), 'w'),
            indent=4, separators=(',', ': ')
        )
        conf = vars(args)
        conf['epoch'] = 0 

    # Define dataset
    print("Defining dataset object ...")
    datasets = []
    dataset_args = eval(args.datasets)
    mean_norm, var_norm = eval(args.mean_var)
    for ds in dataset_args:
        datasets.append(
            HybridAsrDataset(
                ds['data'], ds['tgt'], args.num_targets,
                left_context=ds['left_context'],
                right_context=ds['right_context'],
                chunk_width=ds['chunk_width'],
                batchsize=ds['batchsize'],
                num_split=ds['num_split'],
                validation=args.validation_spks,
                subsample=args.subsample,
                mean=mean_norm, var=var_norm,
            )
        )


    # Get the feature input dimension (needed for the model)
    conf['idim'] = datasets[0].data_shape[1] 

    # Define model
    print("Defining model ...")
    model = models.MODELS[args.model].build_model(conf)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Traning model with ", total_params, " parameters.")
    objective = objectives.OBJECTIVES[args.objective].build_objective(conf)

    # Send model and objective function to GPU (or keep on CPU)
    model.to(device)
    objective.to(device)

    # Define trainable parameters
    params = list(
        filter(
            lambda p: p.requires_grad,
            chain(model.parameters(), objective.parameters()),
        )
    )

    # Define optimizer over trainable parameters and a learning rate schedule
    optimizers = {
        'sgd': torch.optim.SGD(params, lr=conf['lr'], momentum=0.0),
        'adadelta': torch.optim.Adadelta(params, lr=conf['lr']),
        'adam': torch.optim.Adam(params, lr=conf['lr'], weight_decay=conf['weight_decay']),
    }
    
    optimizer = optimizers[conf['optim']]
    lr_sched = LRScheduler(optimizer, args)
   
    # Check if training is resuming from a previous epoch 
    if args.resume is not None:
        print("Resuming ...")
        mdl = torch.load(
            os.path.sep.join([args.expdir, args.resume]),
            map_location=device
        )
        model.load_state_dict(mdl['model'])   
        objective.load_state_dict(mdl['objective'])
        optimizer.load_state_dict(mdl['optimizer'])
        lr_sched.load_state_dict(mdl['lr_sched'])
        conf['epoch'] = mdl['epoch']
    else:
        # Get priors if we are not resuming
        if args.objective == 'CrossEntropy':
            get_priors(args, datasets[0])
    
    # Initializing with a pretrained model
    if args.init is not None:
        mdl = torch.load(args.init, map_location=device)
        for name, p in model.named_parameters():
            if 'xent_layer' not in name and 'linear' not in name: 
                p.data.copy_(mdl['model'][name].data)
  
    # train
    if not args.priors_only:
        train(args, conf, datasets, model, objective, optimizer, lr_sched, device) 
    else:
        datasets[0].utt_subset = [i for i in random.sample(dataset.targets.keys(), 1000)]
        priors = update_priors(args, datasets[0], model, device=device)
        with open(os.path.join(args.expdir, 'priors_updated'), 'w') as f:
            json.dump(priors, f, indent=4, separators=(',', ': '))

    print("Done.")


def train(args, conf, datasets, model, objective, optimizer, lr_sched, device):
    '''
        Runs the training defined in args and conf, on the datasets, using
        the model, objective, optimizeer and lr_scheduler. Performs training
        on the specified device.
    '''
    for e in range(conf['epoch'], conf['epoch'] + args.num_epochs):
        # Train 1 epoch
        generator = multiset_batches(
            args.batches_per_epoch,
            datasets,
        )
        valid_generator = validation_batches(datasets[0]) 
        avg_loss = train_epoch(
            args, generator, model, objective, optimizer, lr_sched, device=device
        )         
        print("Epoch: ", e + 1, " :: Avg Loss =", avg_loss)
        # Run validation set
        if args.objective in ('CrossEntropy'):
            avg_loss_val, avg_acc_val = validate(
                args, valid_generator, model, device=device
            )
            print("Validation Loss: ", avg_loss_val, "Acc: ", avg_acc_val)
        
        # Save checkpoint
        state_dict = {
            'model': model.state_dict(),
            'objective': objective.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_sched': lr_sched.state_dict(),
            'epoch': e + 1,
        }
        
        torch.save(
            state_dict,
            args.expdir + '/{}.{}.mdl'.format(e + 1, args.job),
        )


def get_priors(args, dataset):
    '''
        Compute priors from target defined in the dataset. These are useful
        for training cross-entropy based hybrid HMM-DNN ASR systems as you
        need to compute pseudo-likelihoods from the network posteriors by
        normalizing by these priors. 
    '''
    priors = {i: 1e-13 for i in range(dataset.num_targets)}
    total = sum(priors.values())
    for u in dataset.targets:
        for pdf in dataset.targets[u]:
            priors[pdf] += 1.0
            total += 1.0
    priors_list = []
    C = np.log(total)
    for pdf in sorted(priors.keys()):
        priors_list.append(np.log(priors[pdf]) - C)

    with open(os.path.join(args.expdir, 'priors'), 'w') as f:
        json.dump(priors_list, f, indent=4, separators=(',', ': '))


def update_priors(args, dataset, model, device='cpu'):
    '''
        For a trained model, recompute the priors by using the actual model's
        prediction on a specific dataset. This sometimes improves performance.
    '''
    priors = np.zeros((1, dataset.num_targets)) 
    for u, mat in decode_dataset(args, dataset, model, device=device):
        print('Utt: ', u)
        priors += np.exp(mat).sum(axis=0)
    priors /= priors.sum()
    return np.log(priors).tolist()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',
        default="[{'data': 'data/train_sp_hires',"
            "'tgt': 'data/train_sp_hires/pdfid.tgt',"
            "'batchsize': 128,"
            "'chunk_width': 140,"
            "'left_context': 25,"
            "'right_context': 5}]"
    )
    parser.add_argument('--expdir')
    parser.add_argument('--num-targets', type=int)
    parser.add_argument('--priors-only', action='store_true')
    parser.add_argument('--model', default='TDNN',
        choices=[
            'TDNN',
            'ChainTDNN',
            'Resnet',
            'ChainResnet',
            'WideResnet',
            'ChainWideResnet',
        ]
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--job', type=int, default=1)
    parser.add_argument('--objective', default='CrossEntropy',
        choices=[
            'CrossEntropy',
            'LFMMI',
            'TSComparison',
            'SemisupLFMMI',
            'LFMMI_EBM',
            'CrossEntropy_EBM',
        ],
    )
    parser.add_argument('--subsample', type=int, default=3)
    parser.add_argument('--delay-updates', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batches-per-epoch', type=int, default=500)
    parser.add_argument('--grad-thresh', type=float, default=30.0) 
    parser.add_argument('--optim', default='sgd')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-08)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--init', default=None)

    # Args specific to different components. See model,LRScheduler,dataset}.py.
    args, leftover = parser.parse_known_args()  
    HybridAsrDataset.add_args(parser)
    models.MODELS[args.model].add_args(parser)
    objectives.OBJECTIVES[args.objective].add_args(parser)
    LRScheduler.add_args(parser)
    parser.parse_args(leftover, namespace=args)
    return args


if __name__ == '__main__':
    main()
