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
from copy import deepcopy
import math
from LRScheduler import LRScheduler
from collections import defaultdict
from torch._six import container_abcs


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
        if len(param.size()) > 0: 
            param.mul_(0.0)
    
    fraction = 1.0 / (len(args.models)) 
    for i, m in enumerate(args.models):
        print("Combining Model ", i, " ...")
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
                if len(p.size()) != 0: 
                    new_mdl_dict[name].add_(p, alpha=fraction)
                else:
                    new_mdl_dict[name] = (p * fraction).type(new_mdl_dict[name].dtype)

        #--------------------- Objectives ---------------------
        # To combine objectives is harder: We average parameter weights if
        # applicable, but in the case of some models such as the EBM models
        # we have to specify how to combine things like the sampling buffer.
        # This combination is model specific and should therefore written as a
        # as method in the objective's class. For now we have just done it
        # here though.
        update_opt_state_dict(new_optim_dict, state_dict['optimizer'], fraction) 
        new_objective_dict = objective.add_state_dict(
            new_objective_dict, state_dict['objective'],
            fraction, iteration=i,
        )
        
    new_state_dict = {
        'model': new_mdl_dict,
        'objective': new_objective_dict,
        'optimizer': state_dict['optimizer'],
        'lr_sched': state_dict['lr_sched'],
        'epoch': state_dict['epoch'],
    }
    
    torch.save(
        new_state_dict,
        args.omodel,
    )
    
    if not args.save_models:
        for m in args.models:
            os.remove(m) 


def update_opt_state_dict(state_dict1, state_dict2, fraction):
    '''
        Update state_dict1, with state_dict2 where values are
        val1 + fraction*val2
    '''
    groups2 = state_dict2['param_groups']
    groups1 = state_dict1['param_groups']
    
    if len(groups1) != len(groups2):
        raise ValueError("state dict as a different number of parameter groups")

    param_lens = (len(g['params']) for g in groups1)
    saved_lens = (len(g['params']) for g in groups2)
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError("loaded state dict contains a parameter group that "
            "doesn't match the size of the optimizer's group") 

    id_map = {p: old_id for old_id, p in
        zip(chain(*(g['params'] for g in groups1)),
            chain(*(g['params'] for g in groups2)))}
    
    for k, v in state_dict2['state'].items():
        if k in id_map:
            param = id_map[k]
            if param in state_dict1['state']:
                for p_name, p in v.items():
                    if isinstance(p, torch.Tensor):
                        state_dict1['state'][param][p_name] += fraction * p
            else:
                state_dict1['state'][param] = {key: fraction * val for key, val in v.items()}
        else:
            for p_name, p in v.items():
                if isinstance(p, torch.Tensor):
                    state_dict1['state'][k][p_name] = fraction * p


if __name__ == "__main__":
    main()

