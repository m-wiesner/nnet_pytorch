#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2019  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

from __future__ import print_function
import argparse
import sys
import os
import models
import objectives
import torch
import json
from itertools import chain
import math
from LRScheduler import LRScheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('omodel', help='path to output model', type=str,)
    parser.add_argument('idim', type=int)
    parser.add_argument('conf', type=str)
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--models', nargs='+', type=str, help='paths to models')
    args = parser.parse_args()

    conf = json.load(open(args.conf))
    conf['idim'] = args.idim
    new_model = models.MODELS[conf['model']].build_model(conf)
    objective = objectives.OBJECTIVES[conf['objective']].build_objective(conf)
   
    params = list(
        filter(
            lambda p: p.requires_grad,
            chain(new_model.parameters(), objective.parameters()),
        )
    ) 
   
    optimizers = {
        'sgd': torch.optim.SGD(params, lr=conf['lr'], momentum=0.0),
        'adadelta': torch.optim.Adadelta(params, lr=conf['lr']),
        'adam': torch.optim.Adam(params, lr=conf['lr'], weight_decay=conf['weight_decay']),
    }

    optimizer = optimizers[conf['optim']]
    opt_state_dict = optimizer.state_dict()


    new_mdl_dict = new_model.state_dict()
    new_optim_dict = optimizer.state_dict()
    new_objective_dict = objective.state_dict()

    for name, param in new_mdl_dict.items():
        param.mul_(0.0)

    fraction = 1.0 / (len(args.models)) 
    for i, m in enumerate(args.models):
        state_dict = torch.load(m, map_location=torch.device('cpu'))
        if i == 0 and 'buffer' in state_dict:
            new_buffer = torch.FloatTensor(
                state_dict['buffer'].cpu().size(0),
                state_dict['buffer'].cpu().size(1),
                state_dict['buffer'].cpu().size(2),
            )
            new_buffer_numsteps = torch.zeros(state_dict['buffer'].cpu().size(0))
        
        #----------------------- Model -------------------------
        # To combine models, we just average the weights
        for name, p in state_dict['model'].items():
            if name in new_mdl_dict:
                new_mdl_dict[name].add_(p, alpha=fraction) 
  
        #--------------------- Objectives ---------------------
        # To combine objectives is harder: We average parameter weights if
        # applicable, but in the case of some models such as the EBM models
        # we have to specify how to combine things like the sampling buffer.
        # This combination is model specific and should therefore written as a
        # as method in the objective's class. For now we have just done it
        # here though.
        #for name, p in state_dict['objective'].items():
        #    if name in new_objective_dict:
        #        new_objective_dict[name].add_(fraction, p) 
        
        if 'buffer' in state_dict:
            # Random sample of (fraction * buffersize) indices to take
            buffsize = len(state_dict['buffer'])
            num_samples = math.floor(fraction * buffsize)
            idxs = torch.randint(0, buffsize, num_samples)
            if hasattr(objective, 'sgld_sampler'):
                # Sample fraction of elements from buffer 
                new_buffer[i*num_samples:(i+1)*num_samples] = state_dict['buffer'][idxs].cpu()
                new_buffer_numsteps[i*num_samples:(i+1)*num_samples] = state_dict['buffer_numsteps'][idxs].cpu() 
                state_dict['buffer']
            elif hasattr(objective, 'seq_ebm'):
                new_buffer[i*num_samples:(i+1)*num_samples] = state_dict['buffer'][idxs].cpu() 
                new_buffer_numsteps[i*num_samples:(i+1)*num_samples] = state_dict['buffer_numsteps'][idxs].cpu()
                  
    new_state_dict = {
        'model': new_mdl_dict,
        'objective': new_objective_dict,
        'optimizer': state_dict['optimizer'],
        'lr_sched': state_dict['lr_sched'],
        'epoch': state_dict['epoch'],
    }
    
    if 'buffer' in state_dict:
        new_state_dict = {
            **new_state_dict,
            'buffer': new_buffer,
            'buffer_numsteps': new_buffer_num_steps, 
        }

    torch.save(
        new_state_dict,
        args.omodel,
    )
    
    if not args.save_models:
        for m in args.models:
            os.remove(m) 

if __name__ == "__main__":
    main()

